import os
import numpy as np
import time
import sys


from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet161
from DensenetModels import DenseNet201

from DensenetModels import ResNet152
from DensenetModels import ResNet101
from DensenetModels import ResNet50
from DensenetModels import ResNet34
from DensenetModels import ResNet18
from DRGradeTrainer import DRGradeTrainer

from inferance import DRGradeInferenceMaxMax, DRGradeInferenceMax
from DRTester import DRGradeTesterMaxMax, DRGradeTesterMax
import pandas as pd

DRGradeTrainer = DRGradeTrainer()
DRGradeTesterMaxMax, DRGradeTesterMax  = DRGradeTesterMaxMax(), DRGradeTesterMax()

#-------------------------------------------------------------------------------- 
nclasses = 4
nclasses_expert = 2
def main (nnClassCount, nnIsTrained, IRID_stats=True):
	nnClassCount = nclasses
	nnArchitectureList = [{'name': 'densenet201', 'model' : DenseNet201(nnClassCount, nnIsTrained)}, 
			{'name': 'densenet169', 'model' : DenseNet169(nnClassCount, nnIsTrained)},
			{'name': 'densenet161', 'model' : DenseNet161(nnClassCount, nnIsTrained)}, 
			{'name': 'densenet121', 'model': DenseNet121(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet152', 'model': ResNet152(nnClassCount, nnIsTrained)},
			{'name': 'resnet101', 'model': ResNet101(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet50', 'model': ResNet50(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet34', 'model': ResNet34(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet18', 'model': ResNet18(nnClassCount, nnIsTrained)}]
	#runTest()
	for nnArchitecture in nnArchitectureList:
		runTrain(expert=False, nnArchitecture=nnArchitecture, IRID_stats=IRID_stats)

	nnClassCount = nclasses_expert
	nnArchitectureList = [{'name': 'densenet201', 'model' : DenseNet201(nnClassCount, nnIsTrained)}, 
			{'name': 'densenet169', 'model' : DenseNet169(nnClassCount, nnIsTrained)}, 
			{'name': 'densenet161', 'model' : DenseNet161(nnClassCount, nnIsTrained)}, 
			{'name': 'densenet121', 'model': DenseNet121(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet152', 'model': ResNet152(nnClassCount, nnIsTrained)},
			{'name': 'resnet101', 'model': ResNet101(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet50', 'model': ResNet50(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet34', 'model': ResNet34(nnClassCount, nnIsTrained)}, 
			{'name': 'resnet18', 'model': ResNet18(nnClassCount, nnIsTrained)}]

	for nnArchitecture in nnArchitectureList:
		print ("Expert model training....")
		runTrain(expert=True, nnArchitecture=nnArchitecture, IRID_stats=IRID_stats)


#--------------------------------------------------------------------------------   
def runTrain(expert = True, nnArchitecture = None, IRID_stats=True):
	
	timestampTime = time.strftime("%H%M%S")
	timestampDate = time.strftime("%d%m%Y")
	timestampLaunch = timestampDate + '-' + timestampTime
	
	#---- Path to the directory with images
	if not expert:
		pathTrainData = '../../processed_data/model1/train'
		pathValidData = '../../processed_data/model1/valid'
		nnClassCount = nclasses
	else: 
		pathTrainData = '../../processed_data/expert_model/train'
		pathValidData = '../../processed_data/expert_model/valid'
		nnClassCount = nclasses_expert
	
	#---- Neural network parameters: type of the network, is it pre-trained 
	#---- on imagenet, number of classes
	nnIsTrained = True
	
	#---- Training settings: batch size, maximum number of epochs
	trBatchSize = 16
	trMaxEpoch = 25
	
	#---- Parameters related to image transforms: size of the down-scaled image, cropped image
	imgtransResize = 256
	imgtransCrop = 224
	
	print ('Training NN architecture = ', nnArchitecture)

	DRGradeTrainer.train(pathTrainData, pathValidData, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None, expert, IRID_stats)


#-------------------------------------------------------------------------------- 
def runTest():
	
	pathValidationData = '../../processed_data/train'
	pathTestData = '../../processed_data/test'
	nnIsTrained = True
	Test = True
	MaxMax = True
	Infer = False
	nnClassCount = nclasses
	trBatchSize = 1
	imgtransResize = 256
	imgtransCrop = 224

	pathsModel1 = [
			'../../models/IMAGENET_stat_densenet201.csv',
			'../../models/IMAGENET_stat_densenet169.csv',
			'../../models/IMAGENET_stat_densenet161.csv',
			'../../models/IMAGENET_stat_densenet121.csv',
			'../../models/IMAGENET_stat_resnet152.csv',
			'../../models/IMAGENET_stat_resnet101.csv',
			'../../models/IMAGENET_stat_resnet50.csv',
			'../../models/IMAGENET_stat_resnet34.csv',
			'../../models/IMAGENET_stat_resnet18.csv'

			#'../../models/IDRID_stat_densenet201.csv',
			#'../../models/IDRID_stat_densenet169.csv',
			#'../../models/IDRID_stat_densenet161.csv',
			#'../../models/IDRID_stat_densenet121.csv',
			#'../../models/IDRID_stat_resnet152.csv',
			#'../../models/IDRID_stat_resnet101.csv',
			#'../../models/IDRID_stat_resnet50.csv',
			#'../../models/IDRID_stat_resnet34.csv',
			#'../../models/IDRID_stat_resnet18.csv'
		]

	pathsExpertModel = [
				'../../models/IMAGENET_stat_expert_modeldensenet201.csv',
				'../../models/IMAGENET_stat_expert_modeldensenet169.csv',
				'../../models/IMAGENET_stat_expert_modeldensenet161.csv',
				'../../models/IMAGENET_stat_expert_modeldensenet121.csv',
				'../../models/IMAGENET_stat_expert_modelresnet152.csv',
				'../../models/IMAGENET_stat_expert_modelresnet101.csv',
				'../../models/IMAGENET_stat_expert_modelresnet50.csv',
				'../../models/IMAGENET_stat_expert_modelresnet34.csv',
				'../../models/IMAGENET_stat_expert_modelresnet18.csv'

				#'../../models/IDRID_stat_expert_modeldensenet201.csv',
				#'../../models/IDRID_stat_expert_modeldensenet169.csv',
				#'../../models/IDRID_stat_expert_modeldensenet161.csv',
				#'../../models/IDRID_stat_expert_modeldensenet121.csv',
				#'../../models/IDRID_stat_expert_modelresnet152.csv',
				#'../../models/IDRID_stat_expert_modelresnet101.csv',
				#'../../models/IDRID_stat_expert_modelresnet50.csv',
				#'../../models/IDRID_stat_expert_modelresnet34.csv',
				#'../../models/IDRID_stat_expert_modelresnet18.csv'
			]
		
	timestampLaunch = ''

	# nnArchitecture = DenseNet121(nnClassCount, nnIsTrained)
	print ('Testing the trained model')
	if not Test: path = pathValidationData
	else: path = pathTestData
	
	if MaxMax and not Infer: DRGradeTesterMaxMax.test(Test, path, pathsModel1, pathsExpertModel, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
	elif not MaxMax and not Infer: DRGradeTesterMax.test(Test, path, pathsModel1, pathsExpertModel, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
	elif Infer and not MaxMax: DRGradeInferenceMax.test(Test, path, pathsModel1, pathsExpertModel, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
	else: DRGradeInferenceMaxMax.test(Test, path, pathsModel1, pathsExpertModel, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
#-------------------------------------------------------------------------------- 

if __name__ == '__main__':	
	main(4, True, IRID_stats = True)
	main(4, True, IRID_stats = False) # IMAGENET_stats
	# runTest()
