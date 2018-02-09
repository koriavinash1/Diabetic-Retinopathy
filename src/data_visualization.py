from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
  
def load_resize(base_path, image_ids,  width=224, height=224, resampler_choice=cv2.INTER_AREA, display_images='off'):
	for id in ids:
		path = os.path.join(base_path, id)
		orig_eye_data = np.array(Image.open(path).convert('RGB'))
		gray_eye = cv2.cvtColor(orig_eye_data, cv2.COLOR_RGB2GRAY)

		gray_eye[gray_eye > 20.0] = 255.0

		y, x = np.where(gray_eye == 255.0)
		# print np.min(x), np.max(x)

		# remove background
		eye_data = orig_eye_data[np.min(y):np.max(y), np.min(x):np.max(x)]
		resized_image = cv2.resize(eye_data, (width, height), resampler_choice)

		
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


path ='../raw_data/Training Set/'
ids = next(os.walk(path))[2]
load_resize(path, ids, display_images='Display')





"""
example to see various grades:
Image_id  - DR Grade  -- DME Grade
IDRiD_005 -  4              0
IDRiD_008 -  4              2
IDRiD_010 -  4              1

IDRiD_001 -  3              2
IDRiD_016 -  2              2
IDRiD_021 -  1              0

IDRiD_235 -  0              0

"""
