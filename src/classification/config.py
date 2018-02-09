import numpy as np
import os, cv2

class conf(object):
	def __init__(self):
		self.save_model_path = '../../models/'
		self.log_path = '../../logs'
		self.image_data_path = '../../raw_data/Training Set/'
		self.label_path = '../../raw_data/IDRiD_Training Set.csv'

		self.resize_to = (224, 224)
		self.resampler_choice = cv2.INTER_AREA # bilinear interpolation

		self.data_augmentation = True
		self.input_shape = (224, 224, 3)
		self.resume_training = False
		self.batch_size = 5
		self.epochs = 50
		self.validation_split = 0.05 
		self.learning_rate = 1e-3

		# Set Environment
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'