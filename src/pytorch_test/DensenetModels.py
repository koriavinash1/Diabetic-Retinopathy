import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torchvision.models.densenet import model_urls
from torchvision.models.resnet import model_urls as resnetmodel_urls

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

class DenseNet161(nn.Module):
	
	def __init__(self, classCount, isTrained):
		
		super(DenseNet161, self).__init__()
		model_urls['densenet161'] = model_urls['densenet161'].replace('https://', 'http://')
		self.densenet161 = torchvision.models.densenet161(pretrained=isTrained)
		
		kernelCount = self.densenet161.classifier.in_features
		
		self.densenet161.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
		
	def forward (self, x):
		x = self.densenet161(x)
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

class ResNet152(nn.Module):

	def __init__(self, classCount, isTrained):
	
		super(ResNet152, self).__init__()
		resnetmodel_urls['resnet152'] = resnetmodel_urls['resnet152'].replace('https://', 'http://')
		self.resnet152 = torchvision.models.resnet152(pretrained=isTrained)

		kernelCount = self.resnet152.fc.in_features
		
		self.resnet152.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

	def forward(self, x):
		x = self.resnet152(x)
		return x

class ResNet101(nn.Module):
	
	def __init__(self, classCount, isTrained):
	
		super(ResNet101, self).__init__()
		resnetmodel_urls['resnet101'] = resnetmodel_urls['resnet101'].replace('https://', 'http://')
		self.resnet101 = torchvision.models.resnet101(pretrained=isTrained)

		kernelCount = self.resnet101.fc.in_features
		
		self.resnet101.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

	def forward(self, x):
		x = self.resnet101(x)
		return x
	
class ResNet50(nn.Module):
	
	def __init__(self, classCount, isTrained):
	
		super(ResNet50, self).__init__()
		resnetmodel_urls['resnet50'] = resnetmodel_urls['resnet50'].replace('https://', 'http://')
		self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

		kernelCount = self.resnet50.fc.in_features
		
		self.resnet50.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

	def forward(self, x):
		x = self.resnet50(x)
		return x

class ResNet34(nn.Module):
	
	def __init__(self, classCount, isTrained):
	
		super(ResNet34, self).__init__()
		resnetmodel_urls['resnet34'] = resnetmodel_urls['resnet34'].replace('https://', 'http://')
		self.resnet34 = torchvision.models.resnet34(pretrained=isTrained)

		kernelCount = self.resnet34.fc.in_features
		
		self.resnet34.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

	def forward(self, x):
		x = self.resnet34(x)
		return x
	
class ResNet18(nn.Module):
	
	def __init__(self, classCount, isTrained):
	
		super(ResNet18, self).__init__()
		resnetmodel_urls['resnet18'] = resnetmodel_urls['resnet18'].replace('https://', 'http://')
		self.resnet18 = torchvision.models.resnet18(pretrained=isTrained)

		kernelCount = self.resnet18.fc.in_features
		
		self.resnet18.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

	def forward(self, x):
		x = self.resnet18(x)
		return x
