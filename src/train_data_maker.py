import pandas as pd 
import numpy as np
import os
from sklearn.utils import shuffle
import shutil
from tqdm import tqdm
from PIL import Image
import __future__

#-----------------------------------------------------------------
# Grade 0 : 168	Grade 0 : 134	Grade 0 : 34	
# Grade 1 : 25		Grade 1 : 20	Grade 1 : 05	
# Grade 2 : 168	Grade 2 : 136	Grade 2 : 32	
# Grade 3 : 93		Grade 3 : 74	Grade 3 : 19	 	 	 
# Grade 4 : 62		Grade 4 : 49	Grade 4 : 13
#------------------------------------------------------------------
# additional_images Grade 1 = 89 ; split 65 training : 24 testing
# Total images for training
# Grade 0: 134
# Grade 1: 85 
# Grade 2: 136
# Grade 3: 74 
# Grade 4: 49
#-------------------------------------------------------------------
# Max reshape size = (1024, 1024)
# 
# 
# 
#-----------------------------------------------------------------------


original_full_data_path ='../raw_data/Training Set'
additional_data_path = '../additional_data/images' # retinopathy_grade == 1

csv_path = pd.read_csv('../raw_data/IDRiD_Training Set.csv')

image_name_list = np.array(csv_path['Image name'])
Retinopathy_grade= np.array(csv_path['Retinopathy grade'])

output_path = '../processed_data/train'
if not os.path.exists(output_path):
	os.mkdir(output_path)
else:
	shutil.rmtree(output_path)
	os.mkdir(output_path)



def shrink_image(img_path):
	orig_eye_data = Image.open(img_path).convert('RGB')

	img = np.array(orig_eye_data)
	gray_img = img[:,:,0]
	y, x     = np.where(gray_img>20)
	eye_tight= img[np.min(y):np.max(y),np.min(x):np.max(x)]
	eye_tight= Image.fromarray(eye_tight)
	return eye_tight

for class_of_interest in [0,1,2,3,4]:

	modified_path = output_path+ '/class' + str(class_of_interest) 
	if not os.path.exists (modified_path):
		os.mkdir(modified_path)

	total_num_of_instances = (Retinopathy_grade==class_of_interest).sum()

	image_name_list, Retinopathy_grade = shuffle(image_name_list, Retinopathy_grade, random_state=0)

	modified_image_name_list  = image_name_list[np.where(Retinopathy_grade==class_of_interest)]

	print (len(modified_image_name_list))
	### now let make the first n elements to make the training data

	additional_images = next(os.walk(additional_data_path))[2]

	# add additional data 
	if class_of_interest == 1:
		# add code toresize image
		for img_path in additional_images:
			src = additional_data_path + '/'+ img_path
			dstn = modified_path + '/'+ img_path
			shrink_image(src).save(dstn)

	for i in tqdm(range(total_num_of_instances)):

		src = original_full_data_path + '/'+ modified_image_name_list[i]+'.jpg'
		dstn = modified_path + '/'+ modified_image_name_list[i]+'.jpg'
		
		shrink_image(src).save(dstn)


