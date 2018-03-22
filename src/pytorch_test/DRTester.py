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


class DRGradeTesterMaxMax():
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
		if not expert: path = '../../models/IMAGENET_stat_m-' + name + '.pth.tar'
		else : path = '../../models/IMAGENET_stat_expert-m-' + name + '.pth.tar'

		return path
	
	def accuracy(self, output, labels):
		acc = np.sum(output == labels)/len(labels)
		return float(acc)

	def test (self, Test, pathTestData, pathsModel1, pathsExpertmodel, trBatchSize, transResize, transCrop, launchTimeStamp):
		
		#-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
		IRID_normalize = transforms.Normalize([0.511742964836, 0.243537961753, 0.0797484182405], [0.223165616204, 0.118469339976, 0.0464971614141])
		IMAGENET_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

		#-------------------- SETTINGS: DATASET BUILDERS
		transformList = []
		transformList.append(transforms.Resize(transResize))
		transformList.append(transforms.TenCrop(transCrop))
		transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
		# transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
		transformSequence=transforms.Compose(transformList)
		
		datasetTest = DatasetGenerator(pathImageDirectory=pathTestData, transform=transformSequence)
		dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=1, num_workers=8, shuffle=False, pin_memory=False)
		

		# results per image per model
		_outGTs= torch.FloatTensor().cuda()
		_outPREDs = [] 		
		_image_paths = []
		_mused = []

		# results per image
		outGTs = torch.FloatTensor().cuda()
		outPREDs = [] 		
		image_paths = []

		
		st = time.time() 
		for i, (input, target, path) in enumerate(dataLoaderTest):
			print (path)

			target = target.cuda()
			bs, n_crops, c, h, w = input.size()
			
			max_model1 = []
			for pathModel in pathsModel1:
				if pathModel.__contains__('IRID_stat'): 
					input = transforms.Lambda(lambda crops: torch.stack([IRID_normalize(crop) for crop in crops]))
					#### added by varghese ALEX
					norm_trans= transforms.Normalize([0.511742964836, 0.243537961753, 0.0797484182405], [0.223165616204, 0.118469339976, 0.0464971614141])

					normalized_input = norm_trans(input)



				else: 
					input = transforms.Lambda(lambda crops: torch.stack([IMAGENET_normalize(crop) for crop in crops]))
					### Added by Varghese 
					norm_trans= transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

					normalized_input = norm_trans(input)

				varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)

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

				# logs per model per image
				if class_associated_to_image != 3:
					_outGTs = torch.cat((_outGTs, target), 0)
					_outPREDs.append(class_associated_to_image)
					_image_paths.append(path) 
					_mused.append(pathModel)
				############

			max_primary_output = np.bincount(np.array(max_model1))
			final_output = np.argmax(max_primary_output)

			if final_output == 3:
				max_expert = []
				for pathExpertModel in pathsExpertmodel:
					best_model_path = self.get_best_model_path(pathExpertModel, expert=True)
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

					# logs per model per image
					_outGTs = torch.cat((_outGTs, target), 0)
					_outPREDs.append(class_associated_to_image)
					_image_paths.append(path) 
					_mused.append(pathExpertModel)
					##########
				
				max_expert_output = np.bincount(np.array(max_expert))
				final_output = np.argmax(max_expert_output)

			# per image log...
			outGTs = torch.cat((outGTs, target), 0)
			outPREDs.append(final_output)
			image_paths.append(path)

		outGTs = torch.max(outGTs, 1)[1]
		outGTs = outGTs.cpu().numpy()

		_outGTs = torch.max(_outGTs, 1)[1]
		_outGTs = _outGTs.cpu().numpy()

		# csv creation.....

		# per image csv....
		sub = pd.DataFrame()
		sub['path'] = image_paths
		sub['actual'] = outGTs
		sub['predicted'] = outPREDs
		
		sub.to_csv('../../Testing.csv', index=True)
		

		if not Test:
			# per image per model csv....
			sub = pd.DataFrame()
			sub['path'] = _image_paths
			sub['actual'] = _outGTs
			sub['predicted'] = _outPREDs
			sub['model_used'] = _mused
		
			sub.to_csv('../../Testing_PerModel_PerImage.csv', index=True)
		
		print ("Final Accuracy: {}".format(self.accuracy(outPREDs, outGTs)))
	
#-------------------------------------------------------------------------------- 

class DRGradeTesterMax():
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
		# print (path)
		data = pd.read_csv(path)
		acc = np.squeeze(data['acc'].as_matrix())
		timestamp = np.squeeze(data['timestamp'].as_matrix())
		arch = np.squeeze(data['archs'].as_matrix())

		index = np.where(acc == np.max(acc)) [0][0]
		
		try:
			temp = timestamp[index] + "-" + arch[index]
		except:
			temp = timestamp + "-" + arch

		if not expert: path = '../../models/m-' + temp + '.pth.tar'
		else : path = '../../models/expert-m-' + temp + '.pth.tar'

		return path
	
	def accuracy(self, output, labels):
		acc = np.sum(output == labels)/len(labels)
		return float(acc)

	def test (self,Test, pathTestData, pathsModel1, pathsExpertmodel, trBatchSize, transResize, transCrop, launchTimeStamp):
		
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

		# results per image per model
		_outGTs= torch.FloatTensor().cuda()
		_outPREDs = [] 		
		_image_paths = []
		_mused = []

		# results per image
		outGTs = torch.FloatTensor().cuda()
		outPREDs = [] 		
		image_paths = []
		
		st = time.time() 
		for i, (input, target, path) in enumerate(dataLoaderTest):
			print (path)
			
			target = target.cuda()
			bs, n_crops, c, h, w = input.size()
			varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)

			_outputs = []
			for pathModel in pathsModel1:
				best_model_path = self.get_best_model_path(pathModel, expert=False)
				# print (best_model_path)
				model = torch.load(best_model_path)

				model.eval()			
				out = model(varInput)
				_,class_associated_to_each_crop = torch.max(out,1)
				del _
				class_associated_to_each_crop = class_associated_to_each_crop.data.cpu().numpy() ### numpify
				
				
				count_for_all_classes = np.bincount(class_associated_to_each_crop)

				class_associated_to_image = np.argmax(count_for_all_classes)
				del count_for_all_classes

				# logs per model per image
				if class_associated_to_image != 3:
					_outputs.extend(class_associated_to_each_crop)
					_outGTs = torch.cat((_outGTs, target), 0)
					_outPREDs.append(class_associated_to_image)
					_image_paths.append(path)
					_mused.append(pathModel)
				del class_associated_to_each_crop
				############


				# max_primary_output = np.bincount(np.array(_outputs))
				# final_output = np.argmax(max_primary_output)

				if class_associated_to_image == 3:
					for pathExpertModel in pathsExpertmodel:
						best_model_path = self.get_best_model_path(pathExpertModel, expert=True)
						# print (best_model_path)
						del model
						model = torch.load(best_model_path)
						model.eval()
						out = model(varInput)
						_,class_associated_to_each_crop = torch.max(out,1)
						del _
						class_associated_to_each_crop = class_associated_to_each_crop.data.cpu().numpy() ### numpify
						_outputs.extend(class_associated_to_each_crop + 3)

						count_for_all_classes = np.bincount(class_associated_to_each_crop)
						del class_associated_to_each_crop
						class_associated_to_image_expert_model = np.argmax(count_for_all_classes)
						del count_for_all_classes

						if class_associated_to_image_expert_model ==0:
							class_associated_to_image=3
						else:
							class_associated_to_image=4

						# logs per model per image
						_outGTs = torch.cat((_outGTs, target), 0)
						_outPREDs.append(class_associated_to_image)
						_image_paths.append(path)
						_mused.append(pathExpertModel) 
						##########
								
				
				max_expert_output = np.bincount(np.array(_outputs))
				final_output = np.argmax(max_expert_output)

			# per image log...
			outGTs = torch.cat((outGTs, target), 0)
			outPREDs.append(final_output)
			image_paths.append(path)

		outGTs = torch.max(outGTs, 1)[1]
		outGTs = outGTs.cpu().numpy()

		_outGTs = torch.max(_outGTs, 1)[1]
		_outGTs = _outGTs.cpu().numpy()

		# csv creation.....

		# per image csv....
		sub = pd.DataFrame()
		sub['path'] = image_paths
		sub['actual'] = outGTs
		sub['predicted'] = outPREDs
		
		
		sub.to_csv('../../Testing.csv', index=True)

		
		if not Test:
			# per image per model csv....
			sub = pd.DataFrame()
			sub['path'] = _image_paths
			sub['actual'] = _outGTs
			sub['predicted'] = _outPREDs
			sub['model_used'] = _mused
		
			sub.to_csv('../../Testing_PerModel_PerImage.csv', index=True)

		print ("Final Accuracy: {}".format(self.accuracy(outPREDs, outGTs)))