import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):
	"""docstring for Q_Network"""
	def __init__(self, input_dim=4, output_dim=2, layer_size=[10, 20, 10]):
		super(Q_Network, self).__init__()
		self.input_dim = input_dim
		self.output_dim  = output_dim

		self.layer_size = [input_dim] + layer_size + [output_dim]
		for index in range(len(self.layer_size)-1):
			self.__setattr__('linear%d' % (index+1), nn.Linear(self.layer_size[index], self.layer_size[index+1]))
				
	def forward(self, x):
		out = x
		for index in range(len(self.layer_size)-1):
			out = self.__getattr__('linear%d' % (index+1))(out)
			out = F.relu(out)
			
		return out
