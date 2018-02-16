import pandas as pd 
import numpy as np
import os
from sklearn.utils import shuffle
import shutil
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

# club two different classes
# Retinopathy_grade[Retinopathy_grade == 3] = 3
Retinopathy_grade[Retinopathy_grade == 4] = 3
# If model predicts class 3 image is further sent to expert model
# to classify between 3 and 4

output_path = '../processed_data'
if not os.path.exists(output_path):
	os.mkdir(output_path)
else:
	shutil.rmtree(output_path)
	os.mkdir(output_path)

training_path  = output_path+'/train'
if not os.path.exists(training_path):
	os.mkdir(training_path)
validation_path=output_path+'/valid'
if not os.path.exists(validation_path):
	os.mkdir(validation_path)
testing_path   = output_path+'/test'

if not os.path.exists(testing_path):
	os.mkdir(testing_path)

train_percentage = 0.7
valid_percentage = 0.2
test_percentage  = 0.1

def shrink_image(img_path):
	orig_eye_data = Image.open(img_path).convert('RGB')

	img = np.array(orig_eye_data)
	gray_img = img[:,:,0]
	y, x     = np.where(gray_img>20)
	eye_tight= img[np.min(y):np.max(y),np.min(x):np.max(x)]
	eye_tight= Image.fromarray(eye_tight)
	return eye_tight


# create class wise split and copy data 

for class_of_interest in [0,1,2,3]:

	modified_train_path = training_path+ '/class' + str(class_of_interest) 
	if not os.path.exists (modified_train_path):
		os.mkdir(modified_train_path)

	modified_valid_path = validation_path+ '/class' + str(class_of_interest) 
	if not os.path.exists (modified_valid_path):
		os.mkdir(modified_valid_path)

	modified_test_path = testing_path+ '/class' + str(class_of_interest) 
	if not os.path.exists (modified_test_path):
		os.mkdir(modified_test_path)


	total_num_of_instances = (Retinopathy_grade==class_of_interest).sum()
	train_instance  = int(0.7*total_num_of_instances)
	valid_instance  = int (0.2*total_num_of_instances)
	test_instance   = total_num_of_instances- (train_instance+valid_instance)

	image_name_list, Retinopathy_grade = shuffle(image_name_list, Retinopathy_grade, random_state=0)

	modified_image_name_list  = image_name_list[np.where(Retinopathy_grade==class_of_interest)]

	print len(modified_image_name_list)
	### now let make the first n elements to make the training data

	additional_images = next(os.walk(additional_data_path))[2]
	additional_train_path = additional_images[int(0.7*len(additional_images)):]
	additional_valid_path = additional_images[:int(0.7*len(additional_images))]

	# add additional data 
	if class_of_interest == 1:
		# add code toresize image
		for img_path in additional_train_path:
			src = additional_data_path + '/'+ img_path
			dstn = modified_train_path + '/'+ img_path
			# for additional train
			shrink_image(src).save(dstn)

		for img_path in additional_valid_path:
			src = additional_data_path + '/'+ img_path
			dstn = modified_train_path + '/'+ img_path
			# for additional valid
			shrink_image(src).save(dstn)

		print len(modified_image_name_list) + len(additional_images)

	for i in xrange(train_instance):

		src = original_full_data_path + '/'+ modified_image_name_list[i]+'.jpg'
		dstn = modified_train_path + '/'+ modified_image_name_list[i]+'.jpg'
		
		shrink_image(src).save(dstn)
		# shutil.copy(src,dstn)

	print 'class ' + str(class_of_interest) + ' training done'
	
	for i in range(train_instance,train_instance+valid_instance):
		# print i
		src = original_full_data_path + '/'+ modified_image_name_list[i]+'.jpg'
		dstn = modified_valid_path + '/'+ modified_image_name_list[i]+'.jpg'

		shrink_image(src).save(dstn)
		# shutil.copy(src,dstn)

	print 'class ' + str(class_of_interest) + ' validation done'
	for i in xrange(train_instance+valid_instance,train_instance+valid_instance+test_instance):

		src = original_full_data_path + '/'+ modified_image_name_list[i]+'.jpg'
		dstn = modified_test_path + '/'+ modified_image_name_list[i]+'.jpg'
		
		shrink_image(src).save(dstn)
		# shutil.copy(src,dstn)

	print 'class ' + str(class_of_interest) + ' testing done'