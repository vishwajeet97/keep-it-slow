import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn):
	"""docstring for Q_Network"""
	def __init__(self, input_dim=4, output_dim=2):
		super(Q_Network, self).__init__()
		self.input_dim = input_dim
		self.output_dim  = output_dim
		
		self.linear1 = nn.Linear(input_dim, 10)
		self.linear2 = nn.Linear(10, )

	def forward(self, x):
		out = self.linear1(x)
		out = F.Relu(out)