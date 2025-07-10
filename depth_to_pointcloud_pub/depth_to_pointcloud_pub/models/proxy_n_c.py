import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from torchvision.models.segmentation import deeplabv3_resnet50

def fix_seed(seed=1):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(1)  # 원하는 seed 값

class Proxy(nn.Module):

    def __init__(
            self,
            num_proxies,
            dim,
            num_centers_n=4, alpha=20, temp=10, tau=0.45,
            reg=0.0, lambda_n=0.3, repulsion_weight=0.15):
        super().__init__()
        self.n_proxies = num_proxies # no. of proxies per class
        self.dim = dim
        self.alpha = alpha
        self.print_freq = 40
        self.tau = tau # weight factor for the positive.
        self.reg = reg
        self.temp = temp
        self.lambda_n = lambda_n
        self.repulsion_weight = repulsion_weight
        self.num_centers_n = num_centers_n
        self.momentum = 0.9  # EMA 업데이트용 모멘텀

        self.positive_center = torch.nn.Parameter(torch.randn((1, self.dim), requires_grad=False))
        self.positive_center.data = F.normalize(self.positive_center.data, p=2, dim=1)
        self.proxy_u = torch.nn.Parameter(torch.randn(self.n_proxies, self.dim), requires_grad=True)

        # negative centers (EMA 방식으로 업데이트)
        self.register_buffer('negative_centers', F.normalize(torch.randn(self.num_centers_n, dim), p=2, dim=1))

        # for debugging.
        self.membership_proxies = torch.zeros(self.n_proxies).cuda()
        self.frequency = torch.zeros(self.n_proxies).cuda()
        self.n_features = 0
        self.i=0

        self.eye_idx = (1-torch.eye(self.n_proxies)).bool()
        
    

    def accumulate_membership(self, membership) :

        with torch.no_grad():
            self.n_features += len(membership)
            membership_count = torch.bincount(membership)
            self.membership_proxies[:len(membership_count)] += membership_count.float()

    def reset_membership(self) :

        # Reset Membership tensors
        self.membership_proxies = torch.zeros(self.n_proxies).cuda()
        self.n_features = 0
        
    
    def cosine_similarity(self, feat1, feat2) :
        # (n_a, d), (n_b, d)
        feat1 = F.normalize(feat1, p=2, dim=1, eps=1e-12)
        feat2 = F.normalize(feat2, p=2, dim=1, eps=1e-12)
        similarity = torch.matmul(feat1, feat2.T)

        return similarity

    def update_negative_centers(self, n_feats):
        """
        개선된 negative center 업데이트 (positive와 유사한 feature는 제외하고, 너무 적은 경우 업데이트 생략)
        """
        with torch.no_grad():
            # Step 1: positive center와 너무 유사한 negative feature는 제거
            sim_to_pos = self.cosine_similarity(n_feats, self.positive_center)  # (N, 1)
            keep_mask = (sim_to_pos < 0.7).squeeze()  # cosine similarity threshold
            if keep_mask.sum() == 0:
                return  # 전부 제거되었으면 업데이트 skip

            filtered_feats = n_feats[keep_mask]

            # Step 2: hard assignment
            sim = self.cosine_similarity(filtered_feats, self.negative_centers)  # (N_f, K)
            assign = sim.argmax(dim=1)  # 가장 가까운 negative center 인덱스

            for i in range(self.negative_centers.shape[0]):
                mask = (assign == i)
                if mask.sum() < 3:  # 최소 3개 이상일 때만 업데이트
                    continue
                selected = filtered_feats[mask]
                new_center = F.normalize(selected.mean(dim=0, keepdim=True), p=2, dim=1)
                self.negative_centers[i].lerp_(new_center.squeeze(), 1 - self.momentum)
                self.negative_centers[i] = F.normalize(self.negative_centers[i], p=2, dim=0)

    def inference(self, features_cls) :
            
        # 2. classification.
        dist = self.cosine_similarity(self.positive_center, features_cls.T)

        return dist

    def set_center(self, pos_center) :
        self.positive_center.data = pos_center
    
    # computing codes online.
    @staticmethod
    def sinkhorn(unlabeled_similarity, alpha, n_iterations=5) :
        # unlabeled_similarity : (N, K) (K : # of prototypes, N : # of data points)
        # output : (N, K)

        Q = torch.exp(alpha * unlabeled_similarity).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # import pdb
            # pdb.set_trace()
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment

        return Q.t()
    
    def forward(self, p_feats, u_feats, n_feats=None, pu=False):
        cls_loss = 0
        n_iter = 3
        regularizer = {}
        device = self.positive_center.device

        # 🔹 l_loss: positive feature와 positive center 간의 거리
        if p_feats is None or p_feats.shape[0] == 0:
            l_loss = torch.tensor(0.0, device=device)
        else:
            dist = self.cosine_similarity(self.positive_center, p_feats)
            l_loss = torch.mean(1 - dist)

        # 🔹 n_loss: 확실한 negative와 pseudo negative
        n_loss_total = torch.tensor(0.0, device=device)
        total_count = 0

        if n_feats is not None and n_feats.shape[0] > 0:
            self.update_negative_centers(n_feats)
            sim_n = self.cosine_similarity(n_feats, self.negative_centers)
            target_n = F.softmax(sim_n, dim=1)
            loss_n = -torch.mean(torch.sum(target_n * F.log_softmax(sim_n, dim=1), dim=1))
            n_loss_total += loss_n
            total_count += 1

        with torch.no_grad():
            sim_u_neg = self.cosine_similarity(u_feats, self.negative_centers)
            soft_assign = self.sinkhorn(sim_u_neg.detach(), self.alpha, n_iterations=n_iter)
        logits_u = sim_u_neg
        loss_pseudo = -torch.mean(torch.sum(soft_assign * F.log_softmax(logits_u, dim=1), dim=1))
        n_loss_total += loss_pseudo
        total_count += 1

        if total_count > 0:
            n_loss = n_loss_total / total_count
        else:
            n_loss = torch.tensor(0.0, device=device)

        # 🔹 u_loss: unlabeled feature와 proxy_u 간의 alignment
        if u_feats is None or u_feats.shape[0] == 0:
            u_loss = torch.tensor(0.0, device=device)
        else:
            u_similarity = self.cosine_similarity(u_feats, self.proxy_u)
            with torch.no_grad():
                u_sim_detach = u_similarity.detach()
                u_target = self.sinkhorn(u_sim_detach, self.alpha, n_iterations=n_iter)
            u_loss = -torch.mean(torch.sum(u_target * F.log_softmax(u_similarity * self.temp, dim=1), dim=1))

        # 🔹 repulsion loss: negative center들이 서로 겹치지 않게
        eps = 1e-6
        sim_np = self.cosine_similarity(self.negative_centers, self.positive_center).squeeze(1)
        rep_np = -torch.log(torch.clamp(1.0 - sim_np, min=eps)).mean()

        sim_nn = self.cosine_similarity(self.negative_centers, self.negative_centers)
        mask = ~torch.eye(self.num_centers_n, dtype=bool, device=sim_nn.device)
        rep_nn = -torch.log(torch.clamp(1.0 - sim_nn[mask], min=eps)).mean()

        repulsion_loss = rep_np + 0.1 * rep_nn

        # 🔹 최종 Loss
        cls_loss = (
            self.tau * l_loss +
            self.lambda_n * n_loss +
            (0.55 - self.tau) * u_loss +
            self.repulsion_weight * repulsion_loss
        )

        regularizer['l'] = l_loss
        regularizer['u'] = u_loss
        regularizer['n'] = n_loss
        regularizer['rep'] = repulsion_loss

        return cls_loss, regularizer
