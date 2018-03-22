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

def tight_box(img):
	img = np.array(img)
	y, x     = np.where(img[:,:,0]>20)
	img = img[np.min(y):np.max(y),np.min(x):np.max(x),:]
	img = Image.fromarray(img)
	return img

class DatasetGenerator (Dataset):
	
	#-------------------------------------------------------------------------------- 
	
	def __init__ (self, pathImageDirectory, transform):
	
		self.listImagePaths = []
		self.listImageLabels = []
		self.transform = transform
		
		#---- Open folder get class folder
		try:
			subdir = next(os.walk(pathImageDirectory))[1]
			nclasses = len(subdir)
			print ("Total number of classes: " + str(nclasses))

			for imclass in range(nclasses):
				folder_path = os.path.join(pathImageDirectory, "class"+str(imclass))
				img_paths = next(os.walk(folder_path))[2]

				for img in img_paths:
					img_path = os.path.join(folder_path, img)
					self.listImagePaths.append(img_path)
					self.listImageLabels.append(one_hot(imclass, nclasses))
		except: 
			img_paths = next(os.walk(pathImageDirectory))[2]
			for img in img_paths:
				img_path = os.path.join(folder_path, img)
				self.listImagePaths.append(img_path)

		# self.listImagePaths = self.listImagePaths[:5]
		# self.listImageLabels = self.listImageLabels[:5]
	#-------------------------------------------------------------------------------- 
	
	def __getitem__(self, index):
		
		imagePath = self.listImagePaths[index]
		
		imageData = tight_box(Image.open(imagePath).convert('RGB'))

		try: imageLabel= torch.FloatTensor(self.listImageLabels[index])
		except: imageLabel = None
		
		# print (imageLabel)
		if self.transform != None: imageData = self.transform(imageData)
		
		return imageData, imageLabel, imagePath
		
	#-------------------------------------------------------------------------------- 
	
	def __len__(self):
		
		return len(self.listImagePaths)
	
 #-------------------------------------------------------------------------------- 