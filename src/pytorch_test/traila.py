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
from PIL import Image

from sklearn.metrics.ranking import roc_auc_score

from DatasetGenerator import DatasetGenerator

import torchnet as tnt
import pandas as pd



pathTestData='../../processed_data/test'

IRID_normalize = transforms.Normalize([0.511742964836, 0.243537961753, 0.0797484182405], [0.223165616204, 0.118469339976, 0.0464971614141])
IMAGENET_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#-------------------- SETTINGS: DATASET BUILDERS
datasetTest = DatasetGenerator(pathImageDirectory=pathTestData, transform=None)
dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=1, num_workers=8, shuffle=False, pin_memory=False)

#---------------------custom transforms
transformListIRID = []
transformListIMAGENET = []


transformListIMAGENET.append(transforms.ToPILImage())
transformListIMAGENET.append(transforms.Resize(256))
transformListIMAGENET.append(transforms.TenCrop(224))
transformListIMAGENET.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
transformListIMAGENET.append(transforms.Lambda(lambda crops: torch.stack([IMAGENET_normalize(crop) for crop in crops])))
transformIMAGENET = transforms.Compose(transformListIMAGENET)

imgs, labs, paths = next(iter(dataLoaderTest))
print (imgs.size())

# simgs = Image.fromarray(simgs)
print (transformIMAGENET(imgs.squeeze()).unsqueeze(0).size())
