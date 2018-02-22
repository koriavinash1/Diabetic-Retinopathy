import os
import numpy as np
import time
import sys

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201

from DensenetModels import ResNet152
from DensenetModels import ResNet101
from DensenetModels import ResNet50
from DensenetModels import ResNet34
from DensenetModels import ResNet18
from DRGradeTrainer import DRGradeTrainer
DRGradeTrainer = DRGradeTrainer()

#-------------------------------------------------------------------------------- 
nclasses = 4
nclasses_expert = 2


def main(nnClassCount, nnIsTrained):
	nnArchitectureList = [{'name': 'densenet201', 'model' : DenseNet201(nnClassCount, nnIsTrained)}, 
			{'name': 'densenet169', 'model' : DenseNet169(nnClassCount, nnIsTrained)}, 
			{'name': 'densenet121', 'model': DenseNet121(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet152', 'model': ResNet152(nnClassCount, nnIsTrained)},
			{'name': 'resnet101', 'model': ResNet101(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet50', 'model': ResNet50(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet34', 'model': ResNet34(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet18', 'model': ResNet18(nnClassCount, nnIsTrained)}]
	#runTest()
	for nnArchitecture in nnArchitectureList:
		runTrain(expert=False, nnArchitecture=nnArchitecture)

	nnClassCount = nclasses_expert
	nnArchitectureList = [{'name': 'densenet201', 'model' : DenseNet201(nnClassCount, nnIsTrained)}, 
			{'name': 'densenet169', 'model' : DenseNet169(nnClassCount, nnIsTrained)}, 
			{'name': 'densenet121', 'model': DenseNet121(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet152', 'model': ResNet152(nnClassCount, nnIsTrained)},
			{'name': 'resnet101', 'model': ResNet101(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet50', 'model': ResNet50(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet34', 'model': ResNet34(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet18', 'model': ResNet18(nnClassCount, nnIsTrained)}]

	for nnArchitecture in nnArchitectureList:
		print "Expert model training...."
		runTrain(nnArchitecture=nnArchitecture)
  
#--------------------------------------------------------------------------------   

def runTrain(expert = True, nnArchitecture = None):
	
	timestampTime = time.strftime("%H%M%S")
	timestampDate = time.strftime("%d%m%Y")
	timestampLaunch = timestampDate + '-' + timestampTime
	
	#---- Path to the directory with images
	if not expert:
		pathTrainData = '../../processed_data/train'
		pathValidData = '../../processed_data/valid'
		nnClassCount = nclasses
	else: 
		pathTrainData = '../../processed_data/expert/train'
		pathValidData = '../../processed_data/expert/valid'
		nnClassCount = nclasses_expert
	
	#---- Neural network parameters: type of the network, is it pre-trained 
	#---- on imagenet, number of classes
	nnIsTrained = True
	
	#---- Training settings: batch size, maximum number of epochs
	trBatchSize = 4
	trMaxEpoch = 50
	
	#---- Parameters related to image transforms: size of the down-scaled image, cropped image
	imgtransResize = 256
	imgtransCrop = 224
	
	print ('Training NN architecture = ', nnArchitecture)

	DRGradeTrainer.train(pathTrainData, pathValidData, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None, expert)


#-------------------------------------------------------------------------------- 

def runTest(nnArchitecture):
	
	pathTestData = '../../processed_data/train'
	nnIsTrained = True
	nnClassCount = nclasses
	trBatchSize = 4
	imgtransResize = 256
	imgtransCrop = 224
	
	pathsModel1 = ['../../models/m-densenet201.pth.tar',
			'../../models/m-densenet169.pth.tar',
			'../../models/m-densenet121.pth.tar',
			'../../models/m-resnet152.pth.tar',
			'../../models/m-resnet101.pth.tar',
			'../../models/m-resnet50.pth.tar',
			'../../models/m-resnet34.pth.tar',
			'../../models/m-resnet18.pth.tar']

	pathsExpertModel = ['../../models/expert-m-densenet201.pth.tar',
				'../../models/expert-m-densenet169.pth.tar',
				'../../models/expert-m-densenet121.pth.tar',
				'../../models/expert-m-resnet152.pth.tar',
				'../../models/expert-m-resnet101.pth.tar',
				'../../models/expert-m-resnet50.pth.tar',
				'../../models/expert-m-resnet34.pth.tar',
				'../../models/expert-m-resnet18.pth.tar']
	
	timestampLaunch = ''

	print ('Testing the trained model')
	DRGradeTrainer.test(pathTestData, pathsModel1, pathsExpertModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
	main(nclasses, True)