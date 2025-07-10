# Copyright (c) OpenMMLab. All rights reserved.
import os
# import random
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

def fix_seed(seed=1):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(1)  # 원하는 seed 값

class ContrastiveHead(nn.Module):
	"""Head for contrastive learning.

	The contrastive loss is implemented in this head and is used in SimCLR,
	MoCo, DenseCL, etc.

	Args:
		temperature (float): The temperature hyper-parameter that
			controls the concentration level of the distribution.
			Defaults to 0.1.
	"""

	def __init__(self, temperature=0.1):
		super(ContrastiveHead, self).__init__()
		self.criterion = nn.CrossEntropyLoss()
		self.temperature = temperature

	def cosine_similarity(self, feat1, feat2) :
		# (n_a, d), (n_b, d)
		feat1 = F.normalize(feat1, p=2, dim=1)
		feat2 = F.normalize(feat2, p=2, dim=1)
		similarity = torch.matmul(feat1, feat2.T)

		return similarity

	def create_mask(self, batch_size) :

		pos_ind_1_x = torch.arange(batch_size).cuda()
		pos_ind_1_y = batch_size + torch.arange(batch_size, dtype=torch.long).cuda()
		pos_ind_2_x = batch_size + torch.arange(batch_size).cuda()
		pos_ind_2_y = torch.arange(batch_size, dtype=torch.long).cuda()

		neg_mask = 1 - torch.eye(batch_size * 2, dtype=torch.uint8).cuda()
		neg_mask = neg_mask.bool()
		pos_mask = torch.zeros_like(neg_mask).cuda()

		pos_mask[pos_ind_1_x, pos_ind_1_y] = True
		pos_mask[pos_ind_2_x, pos_ind_2_y] = True
		neg_mask[pos_mask] = False

		return pos_mask, neg_mask

	def forward(self, features_q, features_k, epoch=0) :

		features = torch.cat([features_q, features_k], dim=0)
		similarity = self.cosine_similarity(features, features)
		pos_mask, neg_mask = self.create_mask(len(similarity) // 2)

		pos_similarity = similarity[pos_mask].reshape(-1, 1)
		neg_similarity = similarity[neg_mask].reshape(len(similarity), -1) # (n, n-2)
		loss = self.losses(pos_similarity, neg_similarity)

		return loss

	def losses(self, pos, neg):
		"""Forward function to compute contrastive loss.

		Args:
			pos (Tensor): Nx1 positive similarity.
			neg (Tensor): Nxk negative similarity.

		Returns:
			dict[str, Tensor]: A dictionary of loss components.
		"""
		N = pos.size(0)
		logits = torch.cat((pos, neg), dim=1)
		logits /= self.temperature
		labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)
		losses = self.criterion(logits, labels)
		return losses


class DenseContrastiveHead(nn.Module):
	"""Head for contrastive learning.

	The contrastive loss is implemented in this head and is used in SimCLR,
	MoCo, DenseCL, etc.

	Args:
		temperature (float): The temperature hyper-parameter that
			controls the concentration level of the distribution.
			Defaults to 0.1.
	"""

	def __init__(self, temperature=0.1, coeff=0.5, warmup=0):
		super(DenseContrastiveHead, self).__init__()
		self.criterion = nn.CrossEntropyLoss()
		self.temperature = temperature
		self.coeff = coeff
		self.warmup = warmup

	def cosine_similarity(self, feat1, feat2) :
		# (n_a, d), (n_b, d)
		feat1 = F.normalize(feat1, p=2, dim=1)
		feat2 = F.normalize(feat2, p=2, dim=1)
		similarity = torch.matmul(feat1, feat2.T)

		return similarity

	def create_mask(self, batch_size) :

		pos_ind_1_x = torch.arange(batch_size).cuda()
		pos_ind_1_y = batch_size + torch.arange(batch_size, dtype=torch.long).cuda()
		pos_ind_2_x = batch_size + torch.arange(batch_size).cuda()
		pos_ind_2_y = torch.arange(batch_size, dtype=torch.long).cuda()

		neg_mask = 1 - torch.eye(batch_size * 2, dtype=torch.uint8).cuda()
		neg_mask = neg_mask.bool()
		pos_mask = torch.zeros_like(neg_mask).cuda()

		pos_mask[pos_ind_1_x, pos_ind_1_y] = True
		pos_mask[pos_ind_2_x, pos_ind_2_y] = True
		neg_mask[pos_mask] = False

		return pos_mask, neg_mask

	def forward(self, input_q, input_k, epoch) :

		[original_features_q, avgpooled_global_q, x_q, avgpooled_dense_q] = input_q
		[original_features_k, avgpooled_global_k, x_k, avgpooled_dense_k] = input_k

		original_features = torch.cat([original_features_q, original_features_k], dim=0)
		avgpooled_global = torch.cat([avgpooled_global_q, avgpooled_global_k], dim=0)
		x = torch.cat([x_q, x_k], dim=0)
		avgpooled_dense = torch.cat([avgpooled_dense_q, avgpooled_dense_k], dim=0)
		
		# 1. Global feature loss.
		similarity = self.cosine_similarity(avgpooled_global, avgpooled_global)
		pos_mask, neg_mask = self.create_mask(len(similarity) // 2)

		pos_similarity = similarity[pos_mask].reshape(-1, 1)
		neg_similarity = similarity[neg_mask].reshape(len(similarity), -1) # (n, n-2)
		global_loss = self.losses(pos_similarity, neg_similarity)


		# 2. dense correspondence.

		# pos : from only view of the same image.
		b, d, h, w = original_features.shape
		original_features = original_features.view(b, d, -1)

		# 2.0 normalize l2 to calculate cosine similarity.
		original_features = F.normalize(original_features, p=2, dim=1)
		avgpooled_global = F.normalize(avgpooled_global, p=2, dim=1)
		x = F.normalize(x, p=2, dim=1)
		k_avgpooled_dense = F.normalize(avgpooled_dense, p=2, dim=1)

		# 2.1 split views

		q_original_features = original_features[:b//2]
		k_original_features = original_features[b//2:]
		q_x = x[:b//2]
		k_x = x[b//2:]
		q_avgpooled_dense = avgpooled_dense[:b//2]
		k_avgpooled_dense = avgpooled_dense[b//2:]

		p_dense_1, n_dense_1 = self.dense_losses(q_original_features, q_x, k_original_features, k_x, k_avgpooled_dense, epoch)
		p_dense_2, n_dense_2 = self.dense_losses(k_original_features, k_x, q_original_features, q_x, q_avgpooled_dense, epoch)

		p_dense = torch.cat([p_dense_1, p_dense_2], dim=0)
		n_dense = torch.cat([n_dense_1, n_dense_2], dim=0)
		dense_loss = self.losses(p_dense, n_dense)		

		print(global_loss.item(), dense_loss.item())

		if epoch > self.warmup : 
			loss = self.coeff * global_loss + (1-self.coeff) * dense_loss
		else :
			loss = global_loss
		
		return loss


	def dense_losses(self, q_original_features, q_x, k_original_features, k_x, k_avgpooled_dense, epoch) :
		
		with torch.no_grad():
			backbone_sim_matrix = torch.matmul(q_original_features.permute(0, 2, 1), k_original_features)
			densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]

			# 3. Positive 
			indexed_grid = torch.gather(k_x, dim=2,
										index = densecl_sim_ind.unsqueeze(1).expand(
											-1, k_x.size(1), -1))  # NxCxS^2
		
		densecl_sim_q = (q_x * indexed_grid).sum(1)  # NxS^2

		# # dense positive logits: NS^2X1
		l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)

		# 4. Negative.
		# negative : pooled features of views of different images.
		q_x = q_x.permute(0, 2, 1) # b, n^2, d
		q_x = q_x.reshape(-1, q_x.size(2))
		# dense negative logits: NS^2xK
		l_neg_dense = torch.einsum('nc,ck->nk', [q_x, k_avgpooled_dense.T])

		# 5. mask out view of same images
		neg_mask = torch.ones_like(l_neg_dense)
		n2, b = l_neg_dense.shape
		yy = torch.arange(n2, dtype=torch.long).cuda()
		xx = torch.arange(b, dtype=torch.long).unsqueeze(1).repeat(1, 32*32).view(-1, 1).squeeze()
		neg_mask[yy,xx] = 0
		neg_mask = neg_mask.bool()
		l_neg_dense = l_neg_dense[neg_mask].reshape(n2, -1) # (n2, b-1)

		# if epoch > 30 :
		# 	print(torch.mean(l_pos_dense).item(), torch.mean(l_neg_dense.mean(dim=1)).item())
		# 	pdb.set_trace()


		return l_pos_dense, l_neg_dense

	def losses(self, pos, neg):
		"""Forward function to compute contrastive loss.

		Args:
			pos (Tensor): Nx1 positive similarity.
			neg (Tensor): Nxk negative similarity.

		Returns:
			dict[str, Tensor]: A dictionary of loss components.
		"""
		N = pos.size(0)
		logits = torch.cat((pos, neg), dim=1)
		logits /= self.temperature
		labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)
		losses = self.criterion(logits, labels)
		return losses
