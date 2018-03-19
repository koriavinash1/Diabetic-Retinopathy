import __future__
import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score

from DatasetGenerator import DatasetGenerator

import torchnet as tnt

nclasses = 4
confusion_meter = tnt.meter.ConfusionMeter(nclasses, normalized=True)
from tqdm import tqdm

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transformList = []
transformList.append(transforms.RandomResizedCrop(256))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(normalize)      
transformSequence=transforms.Compose(transformList)

nnClassCount=4
datasetTrain = DatasetGenerator(pathImageDirectory='../../processed_data/train', transform=transformSequence, nclasses=nnClassCount)
# datasetVal =   DatasetGenerator(pathImageDirectory=pathValidData, transform=transformSequence, nclasses = nnClassCount)
	  
dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=16, shuffle=True,  num_workers=8, pin_memory=False)
ip, labs, paths = next(iter(dataLoaderTrain))
# dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=8, pin_memory=False)