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
			nn.LeakyReLU(negative_slope=0.2)
			)
		self.block2 = nn.Sequential(
			nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
			nn.LeakyReLU(negative_slope=0.2)
			)
		self.block3 = nn.Sequential(
			nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
			nn.LeakyReLU(negative_slope=0.2)
			)
		self.block4 = nn.Sequential(
			nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
			nn.LeakyReLU(negative_slope=0.2)
			)
		self.block5 = nn.Sequential(
			nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
			nn.LeakyReLU(negative_slope=0.2)
			)
		self.final = nn.Sequential(
			nn.Conv2d(nc, 3, kernel_size=3, stride=1, padding=1), 
			)
		blocks = [self.block1, self.block2, self.block3, self.block4, self.block5]
		self.blocks = []
		for bb in range(num_blocks):
			self.blocks.append(blocks[bb])
		
	def forward(self, x, new_alpha=None, new_beta=None, new_nu=None):
		orig = x
		for bb in range(len(self.blocks)):
			x = self.blocks[bb](x)
			if torch.isnan(x).any():
				print("layer {} problem".format(bb))
		out = self.final(x)

		if self.mixing:	
			alpha = self.a 
			out = alpha*orig + (1-alpha)*out

		out = torch.max(
			torch.min(out, torch.tensor(1.0).cuda()), 
			torch.tensor(0.0001).cuda()
			)

		return out
