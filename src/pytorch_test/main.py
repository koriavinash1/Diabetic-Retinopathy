import os
import numpy as np
import time
import sys

from DRGradeTrainer import DRGradeTrainer
DRGradeTrainer = DRGradeTrainer()

#-------------------------------------------------------------------------------- 
nclasses = 4
nclasses_expert = 2
def main ():
	
	#runTest()
	runTrain(expert=False)
	print "Expert model training...."
	runTrain()
  
#--------------------------------------------------------------------------------   

def runTrain(expert = True):
	
	DENSENET121 = 'DENSE-NET-121'
	DENSENET169 = 'DENSE-NET-169'
	DENSENET201 = 'DENSE-NET-201'
	
	timestampTime = time.strftime("%H%M%S")
	timestampDate = time.strftime("%d%m%Y")
	timestampLaunch = timestampDate + '-' + timestampTime
	
	#---- Path to the directory with images
	if not expert:
		pathTrainData = '../../processed_data/train'
		pathValidData = '../../processed_data/train'
		nnClassCount = nclasses
	else: 
		pathTrainData = '../../processed_data/expert/train'
		pathValidData = '../../processed_data/expert/train'
		nnClassCount = nclasses_expert
	
	#---- Neural network parameters: type of the network, is it pre-trained 
	#---- on imagenet, number of classes
	nnArchitecture = DENSENET121
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

def runTest():
	
	pathTestData = '../../processed_data/train'
	nnArchitecture = 'DENSE-NET-121'
	nnIsTrained = True
	nnClassCount = nclasses
	trBatchSize = 4
	imgtransResize = 256
	imgtransCrop = 224
	
	pathModel = '../../models/m-25012018-123527.pth.tar'
	
	timestampLaunch = ''

	print ('Testing the trained model')
	DRGradeTrainer.test(pathTestData, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
	main()