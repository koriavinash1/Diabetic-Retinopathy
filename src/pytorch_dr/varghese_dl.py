from PIL import Image
import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
  
def load_resize(base_path, image_ids,  width=224, height=224, resampler_choice=cv2.INTER_AREA, display_images='off'):
	for id in ids:
		path = os.path.join(base_path, id)
		orig_eye_data = np.array(Image.open(path).convert('RGB'))
		gray_eye = orig_eye_data[:,:,0]

		gray_eye[gray_eye > 20.0] = 255.0

		y, x = np.where(gray_eye == 255.0)
		# print np.min(x), np.max(x)

		# remove background
		eye_data = orig_eye_data[np.min(y):np.max(y), np.min(x):np.max(x)]
		eye_datas = Image.fromarray(eye_data)

		resized_image = np.array(eye_datas.resize([224,224],resample=Image.NEAREST))

		
		if display_images == 'Display':
			plt.subplot(1, 3, 1)
			plt.imshow(orig_eye_data)
			plt.title('Original Image')
			plt.subplot(1, 3, 2)
			plt.imshow(eye_data)
			plt.title('Centered Image')
			plt.subplot(1, 3, 3)
			plt.imshow(resized_image)
			plt.title('Resized Image')
			plt.show()
	pass


path ='/home/brats/Diabetic-Retinopathy/raw_data/TrainingSet/'
ids = next(os.walk(path))[2]
load_resize(path, ids, display_images='Display')