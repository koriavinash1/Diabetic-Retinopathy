import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 
def one_hot(val, nclasses):
	a = np.zeros(nclasses)
	a[val] = 1
	return a

class DatasetGenerator (Dataset):
	
	#-------------------------------------------------------------------------------- 
	
	def __init__ (self, pathImageDirectory, transform, nclasses):
	
		self.listImagePaths = []
		self.listImageLabels = []
		self.transform = transform
		self.nclasses = nclasses
	
		#---- Open folder get class folder

		subdir = next(os.walk(pathImageDirectory))[1]
		# assert  self.nclasses == len(subdir),\
		# "nclasses defined is not equal to number of folders in given path" 

		for imclass in range(len(subdir)):
			folder_path = os.path.join(pathImageDirectory, "class"+str(imclass))
			img_paths = next(os.walk(folder_path))[2]

			for img in img_paths:
				img_path = os.path.join(folder_path, img)
				self.listImagePaths.append(img_path)

				self.listImageLabels.append([imclass])
		
		self.listImageLabels = self.listImageLabels[:5]
		self.listImagePaths = self.listImagePaths[:5]
	#-------------------------------------------------------------------------------- 
	
	def __getitem__(self, index):
		
		imagePath = self.listImagePaths[index]
		
		imageData = Image.open(imagePath).convert('RGB')
		imageLabel= torch.LongTensor(self.listImageLabels[index])
		
		# print imagePath, np.array(imageData).shape
		# print  self.listImageLabels[index], imageLabel
		if self.transform != None: imageData = self.transform(imageData)
		
		return imageData, imageLabel, imagePath
		
	#-------------------------------------------------------------------------------- 
	
	def __len__(self):
		
		return len(self.listImagePaths)
	
 #-------------------------------------------------------------------------------- 
	