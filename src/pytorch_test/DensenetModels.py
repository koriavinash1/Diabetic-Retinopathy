import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torchvision.models.densenet import model_urls

import torchvision

class DenseNet121(nn.Module):

	def __init__(self, classCount, isTrained):
	
		super(DenseNet121, self).__init__()
		model_urls['densenet121'] = model_urls['densenet121'].replace('https://', 'http://')
		self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

		kernelCount = self.densenet121.classifier.in_features
		
		self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

	def forward(self, x):
		x = self.densenet121(x)
		return x

class DenseNet169(nn.Module):
	
	def __init__(self, classCount, isTrained):
		
		super(DenseNet169, self).__init__()
		model_urls['densenet169'] = model_urls['densenet169'].replace('https://', 'http://')
		self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)
		
		kernelCount = self.densenet169.classifier.in_features
		
		self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
		
	def forward (self, x):
		x = self.densenet169(x)
		return x
	
class DenseNet201(nn.Module):
	
	def __init__ (self, classCount, isTrained):
		
		super(DenseNet201, self).__init__()
		model_urls['densenet201'] = model_urls['densenet201'].replace('https://', 'http://')
		self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)
		
		kernelCount = self.densenet201.classifier.in_features
		
		self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
		
	def forward (self, x):
		x = self.densenet201(x)
		return x