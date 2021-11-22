import torch
import torch.nn as nn
import torch.nn.functional as F
import random 


class TransNet(nn.Module):
	def __init__(self, nc=18, mixing=True, affine=True, act='lrelu', 
		clamp=True, num_blocks=4, a=0.5):
		super(TransNet, self).__init__() 
		if act == 'relu':
			self.act = nn.ReLU() 
		elif act == 'lrelu':
			self.act = nn.LeakyReLU(negative_slope=0.2)
		elif act == 'tanh':
			self.act = nn.Tanh()
		elif act == 'sigmoid':
			self.act == nn.Sigmoid() 
		# parameters
		self.mixing = mixing 
		self.affine = affine
		self.clamp = clamp
		self.a = a
		#conv blocks
		self.block1 = nn.Sequential(
			nn.Conv2d(3, nc, kernel_size=3, stride=1, padding=1), 
			# nn.BatchNorm2d(nc),
			nn.LeakyReLU(negative_slope=0.2)
			)
		self.block2 = nn.Sequential(
			nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
			# nn.BatchNorm2d(nc),
			nn.LeakyReLU(negative_slope=0.2)
			)
		self.block3 = nn.Sequential(
			nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
			# nn.BatchNorm2d(nc),
			nn.LeakyReLU(negative_slope=0.2)
			)
		self.block4 = nn.Sequential(
			nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
			# nn.BatchNorm2d(nc),
			nn.LeakyReLU(negative_slope=0.2)
			)
		self.block5 = nn.Sequential(
			nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
			# nn.BatchNorm2d(nc),
			nn.LeakyReLU(negative_slope=0.2)
			)
		self.final = nn.Sequential(
			nn.Conv2d(nc, 3, kernel_size=3, stride=1, padding=1), 
			# nn.Sigmoid()
			)
		blocks = [self.block1, self.block2, self.block3, self.block4, self.block5]
		self.blocks = []
		for bb in range(num_blocks):
			self.blocks.append(blocks[bb])
		# if self.mixing:
			# self.alpha = nn.Parameter(self.a*torch.ones(1)) 
			# parameters
			# self.alpha.requires_grad = True 

		# if self.affine:
		# 	self.nu = nn.Parameter(torch.ones((1, 3, 1, 1))) 
		# 	self.nu.requires_grad = True 
		# 	self.beta = nn.Parameter(torch.zeros(1, 3, 1, 1))
		# 	self.beta.requires_grad = True 
		
	def forward(self, x, new_alpha=None, new_beta=None, new_nu=None):
		orig = x
		for bb in range(len(self.blocks)):
			x = self.blocks[bb](x)
			if torch.isnan(x).any():
				print("layer {} problem".format(bb))
			# print(x.shape)
		out = self.final(x)
		# print(x.shape)
	
		# print("alpha:{}, beta:{}, nu:{}".format(alpha, beta, nu))

		if self.mixing:	
			alpha = self.a # random.random()
			# alpha = self.alpha if new_alpha is None else new_alpha 
			out = alpha*orig + (1-alpha)*out
		# if self.affine:
		# 	beta = self.beta if new_beta is None else new_beta
		# 	nu = self.nu if new_nu is None else new_nu
		# 	out = out*nu + beta

		out = torch.max(
			torch.min(out, torch.tensor(1.0).cuda()), 
			torch.tensor(0.0001).cuda()
			)
		# out = torch.clamp(out, min=1e-4, max=1.0)

		return out #, g_out
