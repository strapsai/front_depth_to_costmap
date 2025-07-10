import fastflow_inn as Ff   # FrEIA 대신
import fastflow_inn as Fm 
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import pspnet
from einops import rearrange

import cv2
import random

def fix_seed(seed=1):
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(1)  # 원하는 seed 값

class Normalize(nn.Module) :
    def __init__(self):
        super(Normalize, self).__init__()
    def forward(self, input):
        return F.normalize(input, p=2, dim=1)

def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=(kernel_size-1)//2, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=False),
            Normalize()
        )

    return subnet_conv

def subnet_conv_func_nvp():
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 1, padding=0, bias=False),
            Normalize()
        )

    return subnet_conv

def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=True,
        )
    return nodes

def nf_flow(input_chw, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func_nvp(),
            affine_clamping=clamp,
            permute_soft=True,
        )
    return nodes

class PSPNET_RADIO_SSL(nn.Module):
    def __init__(
        self,
        in_channels=256,
        flow_steps=4,
        freeze_backbone=True,
        flow='fast',
    ):
        super(PSPNET_RADIO_SSL, self).__init__()

        self.global_extractor = pspnet.radio_encoder_ssl(
            pretrained=True,
            zoom_factor=1,
            final_dim=768,
        )

        self.local_extractor = pspnet.pspnet_encoder_ssl(
            pretrained=False,
            zoom_factor=1,
            final_dim=768,
        )

        self.freeze_backbone = freeze_backbone
        self.input_size = (512, 512)

        #######################
        # global + local concat
        concat_dim = 768 * 2
        #######################

        if flow == 'fast':
            self.nf_flows = nf_fast_flow(
                [in_channels, self.input_size[0], self.input_size[1]],
                conv3x3_only=True,
                hidden_ratio=1.0,
                flow_steps=flow_steps,
            )
        elif flow == 'nvp':
            self.nf_flows = nf_flow(
                [in_channels, self.input_size[0], self.input_size[1]],
                flow_steps=flow_steps,
            )

        self.head = NonLinearNeck(
            in_channels=concat_dim,
            hid_channels=512,
            out_channels=in_channels
        )

        # RADIO만 freeze
        for param in self.global_extractor.parameters():
            param.requires_grad = False

        # PSPNet의 백본만 freeze하고, PPM과 cls는 학습되도록 유지
        for name, param in self.local_extractor.named_parameters():
            if name.startswith("layer") or name.startswith("layer0"):
                param.requires_grad = False
            else:
                param.requires_grad = True  # PPM과 cls는 학습됨

        # for name, param in self.local_extractor.named_parameters():
        #         param.requires_grad = True


    def forward(self, inputs, detach=True, dense=False):
        if self.freeze_backbone:
            self.global_extractor.eval()
            self.local_extractor.eval()

        if inputs is None:
            print("Received None inputs in model.")
            return None

        [img_1, mask_1, valid_1], [img_2, mask_2, valid_2] = inputs
        img_1, mask_1, valid_1 = img_1.cuda(), mask_1.cuda(), valid_1.cuda()
        img_2, mask_2, valid_2 = img_2.cuda(), mask_2.cuda(), valid_2.cuda()

        # --- Feature Extraction ---
        global_1 = self.global_extractor(img_1)  # (B, 1024, 32, 32)
        global_2 = self.global_extractor(img_2)

        features_out_1, features_1 = self.local_extractor(img_1, global_1)
        features_out_2, features_2 = self.local_extractor(img_2, global_2)
        # [DEBUG] global_1.shape: torch.Size([8, 1024, 32, 32])
        # [DEBUG] local_1.shape: torch.Size([8, 1024, 32, 32]) 

        features_out = torch.cat([features_out_1, features_out_2], dim=0)
        # [DEBUG] features_out.shape: torch.Size([16, 256, 32, 32])

        mask = torch.cat([mask_1, mask_2], dim=0)
        valid = torch.cat([valid_1, valid_2], dim=0)

        valid_resize = F.interpolate(valid.float(), size=features_out.size()[2:], mode='nearest').bool()[:, 0, :, :]
        mask = mask[:, 0, :, :]
        valid = valid[:, 0, :, :]

        p_mask = valid & (mask == 1)
        n_mask = valid & (mask == 2)
        u_mask = valid & (mask == 0)

        features_out = F.normalize(features_out, p=2, dim=1)
        output, log_jac_dets = self.nf_flows(features_out)

        B, H, W = valid_resize.shape
        log_jac_dets_expanded = log_jac_dets.view(B, 1, 1).expand(B, H, W)
        valid_jac_dets = log_jac_dets_expanded * valid_resize.float()

        output = F.interpolate(output, self.input_size, mode='bilinear', align_corners=False)
        out_flat = output.permute(1, 0, 2, 3).reshape(output.size(1), -1)

        def _gather(mask):
            idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)  # (N,)
            return out_flat.index_select(1, idx)                              # (C,N)

        positive_features  = _gather(p_mask)
        negative_features  = _gather(n_mask)
        unlabeled_features = _gather(u_mask)


        features_1 = self.head(features_1)  # 🔥 features (concat_channel → 256)
        features_2 = self.head(features_2)

        return [positive_features.T, negative_features.T, unlabeled_features.T, valid_jac_dets], [features_1, features_2]
    def inference(self, input_tensor, proxy):
        self.eval()
        if self.freeze_backbone:
            self.global_extractor.eval()
            self.local_extractor.eval()

        with torch.no_grad():
            # 1. Feature 추출
            global_feat = self.global_extractor(input_tensor)  # (B, 1024, 32, 32)
            features, feats = self.local_extractor(input_tensor, global_feat)  # (B, 1024, 32, 32)
            features = F.normalize(features, p=2, dim=1)

            # 3. FastFlow 통과
            output, _ = self.nf_flows(features)  # (B, D, 32, 32)
            output = F.interpolate(output, self.input_size, mode='bilinear', align_corners=False)  # (B, D, 512, 512)

            # 4. Proxy inference
            output = output.squeeze(0)  # (D, 512, 512)
            d, h, w = output.shape
            output_flat = output.view(d, -1)  # (D, H*W)
            similarity_map = proxy.inference(output_flat)  # (H*W)
            similarity_map = similarity_map.view(1, 1, h, w).squeeze(0)  # (1, 1, H, W)

            return similarity_map

    def get_pos_center(self, inputs):
        [img_1, mask_1, valid_1], [img_2, mask_2, valid_2] = inputs
        img_1 = img_1.cuda()
        mask_1 = mask_1.cuda()
        valid_1 = valid_1.cuda()
        img_2 = img_2.cuda()
        mask_2 = mask_2.cuda()
        valid_2 = valid_2.cuda()

        # 1. Feature 추출
        global_feat_1 = self.global_extractor(img_1)
        global_feat_2 = self.global_extractor(img_2)
        local_feat_1 = self.local_extractor(img_1)
        local_feat_2 = self.local_extractor(img_2)

        feat_1 = torch.cat([global_feat_1, local_feat_1], dim=1)
        feat_2 = torch.cat([global_feat_2, local_feat_2], dim=1)

        feat_all = torch.cat([feat_1, feat_2], dim=0)
        mask_all = torch.cat([mask_1, mask_2], dim=0)
        valid_all = torch.cat([valid_1, valid_2], dim=0)

        # 2. resize mask
        mask_all = F.interpolate(mask_all.float(), size=feat_all.size()[2:], mode='bilinear', align_corners=False)
        mask_all = (mask_all >= 0.9).bool()
        valid_all = F.interpolate(valid_all.float(), size=feat_all.size()[2:], mode='nearest').bool()

        p_mask = (mask_all & valid_all)[:, 0, :, :]  # (B, H, W)

        # 3. flow
        feat_all = F.normalize(feat_all, p=2, dim=1)
        output, _ = self.nf_flows(feat_all)  # (B, D, 32, 32)

        output_features = output.transpose(1, 0)  # (D, B, H, W)
        positive_features = output_features[:, p_mask]  # (D, N)
        positive_features = F.normalize(positive_features, p=2, dim=0)
        pos_center = positive_features.mean(dim=1)
        pos_center = F.normalize(pos_center, dim=0)

        return pos_center


class NonLinearNeck(nn.Module):
    """The non-linear neck.

    Structure: fc-bn-[relu-fc-bn] where the substructure in [] can be repeated.
    For the default setting, the repeated time is 1.
    The neck can be used in many algorithms, e.g., SimCLR, BYOL, SimSiam.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of fc layers. Defaults to 2.
        with_bias (bool): Whether to use bias in fc layers (except for the
            last). Defaults to False.
        with_last_bn (bool): Whether to add the last BN layer.
            Defaults to True.
        with_last_bn_affine (bool): Whether to have learnable affine parameters
            in the last BN layer (set False for SimSiam). Defaults to True.
        with_last_bias (bool): Whether to use bias in the last fc layer.
            Defaults to False.
        with_avg_pool (bool): Whether to apply the global average pooling
            after backbone. Defaults to True.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 with_bias=False,
                 with_avg_pool=True
                 ):
        super(NonLinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        # hack: not use bias as it is followed by BN

        self.predictor = nn.Sequential(nn.Linear(in_channels, hid_channels, bias=False),
                                        nn.BatchNorm1d(hid_channels),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hid_channels, out_channels)) # output layer		
    def forward(self, x):

        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)

        return x
