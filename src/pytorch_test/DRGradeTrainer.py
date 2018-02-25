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

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from DatasetGenerator import DatasetGenerator

import torchnet as tnt
import pandas as pd

nclasses = 4
confusion_meter = tnt.meter.ConfusionMeter(nclasses, normalized=True)
from tqdm import tqdm

#-------------------------------------------------------------------------------- 

class DRGradeTrainer ():

	#---- Train the densenet network 
	#---- pathTrainData - path to the directory that contains images
	#---- pathValidData - path to the file that contains image paths and label pairs (training set)
	#---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
	#---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
	#---- nnClassCount - number of output classes 
	#---- trBatchSize - batch size
	#---- trMaxEpoch - number of epochs
	#---- transResize - size of the image to scale down to (not used in current implementation)
	#---- transCrop - size of the cropped image 
	#---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
	#---- checkpoint - if not None loads the model and continues training
	
	def train (self, pathTrainData, pathValidData, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint, expert = False):

		
		#-------------------- SETTINGS: NETWORK ARCHITECTURE
		model = nnArchitecture['model']
		
		model = torch.nn.DataParallel(model)
				
		#-------------------- SETTINGS: DATA TRANSFORMS
		normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		
		transformList = []
		transformList.append(transforms.Resize(transResize))
		transformList.append(transforms.RandomResizedCrop(transCrop))
		transformList.append(transforms.RandomHorizontalFlip())
		transformList.append(transforms.ToTensor())
		transformList.append(normalize)      
		transformSequence=transforms.Compose(transformList)

		#-------------------- SETTINGS: DATASET BUILDERS
		datasetTrain = DatasetGenerator(pathImageDirectory=pathTrainData, transform=transformSequence)
		datasetVal =   DatasetGenerator(pathImageDirectory=pathValidData, transform=transformSequence)
			  
		dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=8, pin_memory=False)
		dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=8, pin_memory=False)
		
		# print len(dataLoaderTrain), len(datasetTrain)
		#-------------------- SETTINGS: OPTIMIZER & SCHEDULER
		optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
		scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
				
		#-------------------- SETTINGS: LOSS
		loss = torch.nn.BCELoss()
		# loss = torch.nn.CrossEntropyLoss()
		
		#---- Load checkpoint 
		if checkpoint != None:
			modelCheckpoint = torch.load(checkpoint)
			model.load_state_dict(modelCheckpoint['state_dict'])
			optimizer.load_state_dict(modelCheckpoint['optimizer'])

		
		#---- TRAIN THE NETWORK
		
		lossMIN = 100000
		sub = pd.DataFrame()

		timestamps = []
		archs = []
		losses = []
		accs = []
		for epochID in range (0, trMaxEpoch):
			
			timestampTime = time.strftime("%H%M%S")
			timestampDate = time.strftime("%d%m%Y")
			timestampSTART = timestampDate + '-' + timestampTime
			
			print (str(epochID)+"/" + str(trMaxEpoch) + "---")
			self.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, trBatchSize)
			lossVal, losstensor, acc = self.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, trBatchSize)
			
			timestampTime = time.strftime("%H%M%S")
			timestampDate = time.strftime("%d%m%Y")
			launchTimestamp = timestampDate + '-' + timestampTime
			
			scheduler.step(losstensor.data[0])
			
			if lossVal < lossMIN:
				lossMIN = lossVal
				timestamps.append(launchTimestamp)
				archs.append(nnArchitecture['name'])
				losses.append(lossMIN)
				accs.append(acc)
				if not expert: model_name = '../../models/m-' + launchTimestamp + "-"+ nnArchitecture['name'] + '.pth.tar'
				else : model_name = '../../models/expert-m-' + launchTimestamp + "-" + nnArchitecture['name'] + '.pth.tar'

				torch.save(model, model_name)
				print ('Epoch [' + str(epochID + 1) + '] [save] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' accuracy= ' + str(acc))
				# print confusion_meter
			else:
				print ('Epoch [' + str(epochID + 1) + '] [----] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' accuracy= ' + str(acc))
		sub['timestamp'] = timestamps
		sub['archs'] = archs
		sub['loss'] = losses
		sub['acc'] = accs

		if not expert: sub.to_csv('../../models/' + nnArchitecture['name'] + '.csv', index=True)
		else: sub.to_csv('../../models/' + 'expert_model' + nnArchitecture['name'] + '.csv', index=True)
		
					 
	#-------------------------------------------------------------------------------- 

	# compute accuracy
	def accuracy(self, output, labels):
		pred = torch.max(output, 1)[1]
		label = torch.max(labels, 1)[1]
		acc = torch.sum(pred == label)
		return float(acc)
	

	#--------------------------------------------------------------------------------
	def epochTrain (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):
		
		model.train()
		for batchID, (input, target, _) in tqdm(enumerate (dataLoader)):
						
			target = target.cuda(async = True)
			
			varInput = torch.autograd.Variable(input.cuda())
			varTarget = torch.autograd.Variable(target)         
			varOutput = model(varInput)

			lossvalue = loss(varOutput, varTarget)
			# for crossentropy loss
			# varTarget = torch.max(varOutput, 1)[1]
			# varOutput = torch.max(varOutput, 1)[1]
			# lossvalue = loss(varOutput, varTarget)
			# print "BatchID: {}".format(batchID) + " loss :{}".format(lossvalue.data)		   
			optimizer.zero_grad()
			lossvalue.backward()
			optimizer.step()
			
	#-------------------------------------------------------------------------------- 
		
	def epochVal (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):
		
		model.eval ()
		
		lossVal = 0
		lossValNorm = 0
		
		losstensorMean = 0
		confusion_meter.reset()

		acc = 0.0
		for i, (input, target, _) in enumerate (dataLoader):
			
			target = target.cuda(async=True)
				 
			varInput = torch.autograd.Variable(input.cuda(), volatile=True)
			varTarget = torch.autograd.Variable(target, volatile=True)    
			varOutput = model(varInput)
			
			acc += self.accuracy(varOutput, varTarget)/ (len(dataLoader)*trBatchSize)
			losstensor = loss(varOutput, varTarget)
			
			# for crossentropy loss
			# varTarget = torch.max(varOutput, 1)[1]
			# varOutput = torch.max(varOutput, 1)[1]
			# losstensor= loss(varOutput, varTarget)

			losstensorMean += losstensor
			# confusion_meter.add(varOutput.view(-1), varTarget.data.view(-1))
			lossVal += losstensor.data[0]
			lossValNorm += 1

		
		outLoss = lossVal / lossValNorm
		losstensorMean = losstensorMean / lossValNorm
		
		return outLoss, losstensorMean, acc