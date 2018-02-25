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
import pandas as pd

nclasses = 4
confusion_meter = tnt.meter.ConfusionMeter(nclasses, normalized=True)
from tqdm import tqdm

class DRGradeTester():
	#--------------------------------------------------------------------------------  
	#---- Test the trained network 
	#---- pathDirData - path to the directory that contains images
	#---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
	#---- pathFileVal - path to the file that contains image path and label pairs (validation set)
	#---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
	#---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
	#---- nnClassCount - number of output classes 
	#---- trBatchSize - batch size
	#---- trMaxEpoch - number of epochs
	#---- transResize - size of the image to scale down to (not used in current implementation)
	#---- transCrop - size of the cropped image 
	#---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
	#---- checkpoint - if not None loads the model and continues training

	def get_best_model_path(self, path, expert = True):
		data = pd.read_csv(path)
		acc = np.squeeze(data['acc'].as_matrix())
		timestamp = np.squeeze(data['timestamp'].as_matrix())
		arch = np.squeeze(data['archs'].as_matrix())

		index = np.where(acc == np.max(acc)) [0][0]

		if not expert: path = '../../models/m-' + timestamp[index] + "-" + arch[index] + '.pth.tar'
		else : path = '../../models/expert-m-' + timestamp[index] + "-" + arch[index] + '.pth.tar'

		return path
	
	def accuracy(self, output, labels):
		acc = np.sum(output == labels)/len(labels)
		return float(acc)

	def test (self, pathTestData, pathsModel1, pathsExpertmodel, trBatchSize, transResize, transCrop, launchTimeStamp):
		
		#-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
		normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		
		#-------------------- SETTINGS: DATASET BUILDERS
		transformList = []
		transformList.append(transforms.Resize(transResize))
		transformList.append(transforms.TenCrop(transCrop))
		transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
		transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
		transformSequence=transforms.Compose(transformList)
		
		datasetTest = DatasetGenerator(pathImageDirectory=pathTestData, transform=transformSequence)
		dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=1, num_workers=8, shuffle=False, pin_memory=False)

		outGT = torch.FloatTensor().cuda()
		# outPRED = torch.FloatTensor().cuda()
		outPRED=[]
		paths = []
		
		for i, (input, target, path) in enumerate(dataLoaderTest):
			print (path)
			target = target.cuda()
			paths.append(path)
			outGT = torch.cat((outGT, target), 0)
			
			class_pred = []
			for pathModel, pathExpertmodel in zip(pathsModel1, pathsExpertmodel):
				best_model_path = self.get_best_model_path(pathModel, expert=False)
				print (best_model_path)
				model = torch.load(best_model_path)

				model.eval()
	
				bs, n_crops, c, h, w = input.size()
				varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)
			
				out = model(varInput)
				# print (out)
				_,class_associated_to_each_crop = torch.max(out,1)
				# print ()
				# print (class_associated_to_each_crop)
				class_associated_to_each_crop = class_associated_to_each_crop.data.cpu().numpy() ### numpify
				count_for_all_classes = np.bincount(class_associated_to_each_crop)
				class_associated_to_image = np.argmax(count_for_all_classes)
				# class_one_prediction  = class_associated_to_each_crop.count(1)
				# class_two_prediction  = class_associated_to_each_crop.count(2)
				# class_three_prediction= class_associated_to_each_crop.count(3)

				# print (class_associated_to_image)
				# print (class_associated_to_each_crop)
				# outMean = out.view(bs, n_crops, -1).mean(1)
				# print (outMean)
				# _,class_associated = torch.max(outMean,1)
				# print (class_associated)
				if class_associated_to_image == 3:
					best_expert_path = self.get_best_model_path(pathExpertmodel, expert=True)
					del model
					model = torch.load(best_expert_path)
					model.eval()

					out = model(varInput)
					_,class_associated_to_each_crop = torch.max(out,1)
					class_associated_to_each_crop = class_associated_to_each_crop.data.cpu().numpy() ### numpify
					count_for_all_classes = np.bincount(class_associated_to_each_crop)
					class_associated_to_image_expert_model = np.argmax(count_for_all_classes)
					if class_associated_to_image_expert_model ==0:
						class_associated_to_image=3
					else:
						class_associated_to_image=4

	
				# class_associated_to_image = torch.from_numpy(class_associated_to_image)
				class_pred = np.append(class_pred,class_associated_to_image)

			count_= np.bincount(np.array(class_pred, dtype='int32'))
			outPRED = np.append(outPRED, np.argmax(count_))
		# print (outGT, torch.max(outGT, 1))
		outGT = torch.max(outGT, 1)[1]
		outGT = outGT.cpu().numpy()
		print (outGT.shape, len(paths), outPRED.shape)
		sub = pd.DataFrame()
		sub['path'] = paths
		sub['actual'] = outGT
		sub['predicted'] = outPRED
		
		sub.to_csv('../../Testing.csv', index=True)
		acc = self.accuracy(outPRED, outGT)
		print (acc)

	 
		# return
#-------------------------------------------------------------------------------- 