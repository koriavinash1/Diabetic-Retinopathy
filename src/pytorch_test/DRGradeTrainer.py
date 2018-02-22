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
		print model
		model = torch.nn.DataParallel(model)
				
		#-------------------- SETTINGS: DATA TRANSFORMS
		normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		
		transformList = []
		transformList.append(transforms.RandomResizedCrop(transCrop))
		transformList.append(transforms.RandomHorizontalFlip())
		transformList.append(transforms.ToTensor())
		transformList.append(normalize)      
		transformSequence=transforms.Compose(transformList)

		#-------------------- SETTINGS: DATASET BUILDERS
		datasetTrain = DatasetGenerator(pathImageDirectory=pathTrainData, transform=transformSequence, nclasses=nnClassCount)
		datasetVal =   DatasetGenerator(pathImageDirectory=pathValidData, transform=transformSequence, nclasses = nnClassCount)
			  
		dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=8, pin_memory=False)
		dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=8, pin_memory=False)
		
		# print len(dataLoaderTrain), len(datasetTrain)
		#-------------------- SETTINGS: OPTIMIZER & SCHEDULER
		optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
		scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
				
		#-------------------- SETTINGS: LOSS
		# loss = torch.nn.BCELoss()
		criterion = torch.nn.CrossEntropyLoss()
		
		#---- Load checkpoint 
		if checkpoint != None:
			modelCheckpoint = torch.load(checkpoint)
			model.load_state_dict(modelCheckpoint['state_dict'])
			optimizer.load_state_dict(modelCheckpoint['optimizer'])

		
		#---- TRAIN THE NETWORK
		
		lossMIN = 100000
		
		for epochID in range (0, trMaxEpoch):
			
			timestampTime = time.strftime("%H%M%S")
			timestampDate = time.strftime("%d%m%Y")
			timestampSTART = timestampDate + '-' + timestampTime
			
			print str(epochID)+"/" + str(trMaxEpoch) + "---"
			self.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, criterion, trBatchSize)
			lossVal, losstensor, acc = self.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, criterion, trBatchSize)
			
			timestampTime = time.strftime("%H%M%S")
			timestampDate = time.strftime("%d%m%Y")
			timestampEND = timestampDate + '-' + timestampTime
			
			scheduler.step(losstensor.data[0])
			
			if lossVal < lossMIN:
				lossMIN = lossVal
				# if not expert: model_name = '../../models/m-' + launchTimestamp + '.pth.tar'
				# else : model_name = '../../models/expert-m-' + launchTimestamp + '.pth.tar'

				if not expert: model_name = '../../models/m-' + nnArchitecture['name'] + '.pth.tar'
				else : model_name = '../../models/expert-m-' + nnArchitecture['name'] + '.pth.tar'

				# torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, model_name)
				torch.save(model, model_name)
				print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal)) + ' accuracy= ' + str(acc)
				# print confusion_meter
			else:
				print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal)) + ' accuracy= ' + str(acc)
					 
	#-------------------------------------------------------------------------------- 

	# compute accuracy
	def accuracy(self, output, labels):
		pred = torch.max(output, 1)[1]
		# label = torch.max(labels, 1)[1]
		acc = torch.sum(pred == labels)
		return float(acc)
	

	#--------------------------------------------------------------------------------
	def epochTrain (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, criterion, trBatchSize):
		
		model.train()
		for batchID, (input, target, _) in tqdm(enumerate (dataLoader)):
						
			# target = target.cuda(async = True)
			
			varInput = torch.autograd.Variable(input)
			varTarget = torch.squeeze(torch.autograd.Variable(target))
			varOutput = model(varInput)

			# weighted cross entropy
			eps = 1e-5
			numpified_labels = varTarget.cpu().data.numpy()
			w0 = (numpified_labels==0).sum() + eps
			w1 = (numpified_labels==1).sum() + eps
			w2 = (numpified_labels==2).sum() + eps
			w3 = (numpified_labels==3).sum() + eps
			# w4 = (numpified_labels==4).sum()

			wce =  torch.from_numpy(np.asarray([1/w0, 1/w1, 1/w2, 1/w3])).float()
			# criterion = torch.nn.CrossEntropyLoss(weight=wce)

			lossvalue = criterion(varOutput, varTarget)
			lossvalue.backward()
			optimizer.step()
			
	#-------------------------------------------------------------------------------- 
		
	def epochVal (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, criterion, trBatchSize):
		
		model.eval ()
		
		lossVal = 0
		lossValNorm = 0
		
		losstensorMean = 0
		confusion_meter.reset()
		outputs = []
		targets = []

		acc = 0.0
		for i, (input, target, _s) in enumerate (dataLoader):
			
			# target = target.cuda(async=True)
				 
			varInput = torch.autograd.Variable(input, volatile=True)
			varTarget = torch.squeeze(torch.autograd.Variable(target, volatile=True) )
			varOutput = model(varInput)
			outputs.extend(varOutput.data)
			targets.extend(varTarget.data)

			acc += self.accuracy(varOutput, varTarget)/ (len(dataLoader)*trBatchSize)
			# confusion_meter.conf += 
			losstensor = criterion(varOutput, varTarget)

			# for crossentropy loss
			# varTarget = torch.max(varOutput, 1)[1]
			# varOutput = torch.max(varOutput, 1)[1]
			# losstensor= loss(varOutput, varTarget)

			losstensorMean += losstensor
			lossVal += losstensor.data[0]
			lossValNorm += 1

		outputs = torch.from_numpy(np.array(outputs))
		targets = torch.from_numpy(np.array(targets))

		_, preds = torch.max(outputs.data, 1)
		del _ 
		confusion_meter.add(preds.view(-1), targets.data.view(-1))

		print confusion_meter.conf

		outLoss = lossVal / lossValNorm
		losstensorMean = losstensorMean / lossValNorm
		
		return outLoss, losstensorMean, acc