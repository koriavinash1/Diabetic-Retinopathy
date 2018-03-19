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
import time 


class DRGradeInferenceMaxMax():
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

		index = np.where(acc == np.max(acc)) [0]
		index = index[len(index) - 1]
		try: 
			name = timestamp[index] + "-" + arch[index]
		except: 
			name = timestamp + "-" + arch
		if not expert: path = '../../models/m-' + name + '.pth.tar'
		else : path = '../../models/expert-m-' + name + '.pth.tar'

		return path
	
	def accuracy(self, output, labels):
		acc = np.sum(output == labels)/len(labels)
		return float(acc)

	def test (self, Test, pathTestData, pathsModel1, pathsExpertmodel, trBatchSize, transResize, transCrop, launchTimeStamp):
		
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
		
		outPREDs = [] 		
		image_paths = []
		
		st = time.time() 
		for i, (input, _, path) in enumerate(dataLoaderTest):
			print (path)

			bs, n_crops, c, h, w = input.size()
			varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)

			max_model1 = []
			for pathModel in pathsModel1:
				best_model_path = self.get_best_model_path(pathModel, expert=False)
				print (best_model_path)
				model = torch.load(best_model_path)

				model.eval()			
				out = model(varInput)
				_,class_associated_to_each_crop = torch.max(out,1)
				del _

				class_associated_to_each_crop = class_associated_to_each_crop.data.cpu().numpy() ### numpify
				count_for_all_classes = np.bincount(class_associated_to_each_crop)
				del class_associated_to_each_crop
				class_associated_to_image = np.argmax(count_for_all_classes)
				del count_for_all_classes

				max_model1.append(class_associated_to_image)

			max_primary_output = np.bincount(np.array(max_model1))
			final_output = np.argmax(max_primary_output)

			if final_output == 3:
				max_expert = []
				for pathExpertmodel in pathsExpertmodel:
					best_model_path = self.get_best_model_path(pathExpertmodel, expert=True)
					print (best_model_path)
					del model
					model = torch.load(best_model_path)
					model.eval()
					out = model(varInput)
					_,class_associated_to_each_crop = torch.max(out,1)
					del _
					class_associated_to_each_crop = class_associated_to_each_crop.data.cpu().numpy() ### numpify

					count_for_all_classes = np.bincount(class_associated_to_each_crop)
					del class_associated_to_each_crop
					class_associated_to_image_expert_model = np.argmax(count_for_all_classes)
					if class_associated_to_image_expert_model ==0:
						class_associated_to_image=3
					else:
						class_associated_to_image=4

					max_expert.append(class_associated_to_image)
				
				max_expert_output = np.bincount(np.array(max_expert))
				final_output = np.argmax(max_expert_output)

			image_paths.append(path)
			outPREDs.append(final_output)
		
		sub = pd.DataFrame()
		sub['ImagePaths'] = image_paths
		sub['predicted'] = outPREDs
		
		sub.to_csv('../../Inference.csv', index=True)
		print("time: {}".format(time.time()-st))
	 
		# return
#-------------------------------------------------------------------------------- 

class DRGradeInferenceMax():
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

		outPREDs = [] 		
		image_paths = []
		

		for i, (input, _, path) in enumerate(dataLoaderTest):
			print (path)
			

			bs, n_crops, c, h, w = input.size()
			varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)

			primary_outputs = []
			for pathModel in pathsModel1:
				best_model_path = self.get_best_model_path(pathModel, expert=False)
				print (best_model_path)
				model = torch.load(best_model_path)

				model.eval()			
				out = model(varInput)
				_,class_associated_to_each_crop = torch.max(out,1)
				del _
				class_associated_to_each_crop = class_associated_to_each_crop.data.cpu().numpy() ### numpify
				primary_outputs.extend(class_assocoated_to_each_crop)
				
				count_for_all_classes = np.bincount(class_associated_to_each_crop)
				del class_associated_to_each_crop
				class_associated_to_image = np.argmax(count_for_all_classes)
				del count_for_all_classes


			max_primary_output = np.bincount(np.array(primary_outputs))
			final_output = np.argmax(max_primary_output)

			if final_output == 3:
				expert_outputs = []
				for pathExpertmodel in pathsExpertmodel:
					best_model_path = self.get_best_model_path(pathExpertmodel, expert=True)
					print (best_model_path)
					del model
					model = torch.load(best_model_path)
					model.eval()
					out = model(varInput)
					_,class_associated_to_each_crop = torch.max(out,1)
					del _
					class_associated_to_each_crop = class_associated_to_each_crop.data.cpu().numpy() ### numpify
					expert_outputs.extend(class_associated_to_each_crop)

					count_for_all_classes = np.bincount(class_associated_to_each_crop)
					del class_associated_to_each_crop
					class_associated_to_image_expert_model = np.argmax(count_for_all_classes)
					del count_for_all_classes

					if class_associated_to_image_expert_model ==0:
						class_associated_to_image=3
					else:
						class_associated_to_image=4
								
				
				max_expert_output = np.bincount(np.array(expert_outputs))
				final_output = np.argmax(max_expert_output)
			

			image_paths.append(path)
			outPREDs.append(final_output)



		print("time: {}".format(time.time()-st))

