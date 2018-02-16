import pandas as pd 
import numpy as np
import os
from sklearn.utils import shuffle
import shutil
from PIL import Image

original_full_data_path ='../raw_data/Training Set'

csv_path = pd.read_csv('../raw_data/IDRiD_Training Set.csv')
image_name_list = np.array(csv_path['Image name'])
Retinopathy_grade= np.array(csv_path['Retinopathy grade'])
Retinopathy_grade[Retinopathy_grade == 3] = 2
Retinopathy_grade[Retinopathy_grade == 4] = 3

output_path = '../processed_data'
if not os.path.exists(output_path):
	os.mkdir(output_path)

training_path  =output_path+'/train'
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

for class_of_interest in [0,1,2,3]:
	if class_of_interest ==0:

		modified_train_path = training_path+ '/class0'
		if not os.path.exists (modified_train_path):
			os.mkdir(modified_train_path)


		modified_valid_path = validation_path+ '/class0'
		if not os.path.exists (modified_valid_path):
			os.mkdir(modified_valid_path)
	
		modified_test_path = testing_path+ '/class0'
		if not os.path.exists (modified_test_path):
			os.mkdir(modified_test_path)


	if class_of_interest ==1:

		modified_train_path = training_path+ '/class1'
		if not os.path.exists (modified_train_path):
			os.mkdir(modified_train_path)


		modified_valid_path = validation_path+ '/class1'
		if not os.path.exists (modified_valid_path):
			os.mkdir(modified_valid_path)
	
		modified_test_path = testing_path+ '/class1'
		if not os.path.exists (modified_test_path):
			os.mkdir(modified_test_path)

	if class_of_interest ==2:

		modified_train_path = training_path+ '/class2'
		if not os.path.exists (modified_train_path):
			os.mkdir(modified_train_path)


		modified_valid_path = validation_path+ '/class2'
		if not os.path.exists (modified_valid_path):
			os.mkdir(modified_valid_path)
	
		modified_test_path = testing_path+ '/class2'
		if not os.path.exists (modified_test_path):
			os.mkdir(modified_test_path)


	if class_of_interest ==3:

		modified_train_path = training_path+ '/class3'
		if not os.path.exists (modified_train_path):
			os.mkdir(modified_train_path)


		modified_valid_path = validation_path+ '/class3'
		if not os.path.exists (modified_valid_path):
			os.mkdir(modified_valid_path)
	
		modified_test_path = testing_path+ '/class3'
		if not os.path.exists (modified_test_path):
			os.mkdir(modified_test_path)

	if class_of_interest ==4:

		modified_train_path = training_path+ '/class4'
		if not os.path.exists (modified_train_path):
			os.mkdir(modified_train_path)


		modified_valid_path = validation_path+ '/class4'
		if not os.path.exists (modified_valid_path):
			os.mkdir(modified_valid_path)
	
		modified_test_path = testing_path+ '/class4'
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



	for i in xrange(train_instance):

		src = original_full_data_path + '/'+ modified_image_name_list[i]+'.jpg'
		dstn = modified_train_path + '/'+ modified_image_name_list[i]+'.jpg'
		
		shrink_image(src).save(dstn)
		# shutil.copy(src,dstn)

	print 'training done'
	
	for i in range(train_instance,train_instance+valid_instance):
		# print i
		src = original_full_data_path + '/'+ modified_image_name_list[i]+'.jpg'
		dstn = modified_valid_path + '/'+ modified_image_name_list[i]+'.jpg'

		shrink_image(src).save(dstn)
		# shutil.copy(src,dstn)

	print 'validation done'
	for i in xrange(train_instance+valid_instance,train_instance+valid_instance+test_instance):

		src = original_full_data_path + '/'+ modified_image_name_list[i]+'.jpg'
		dstn = modified_test_path + '/'+ modified_image_name_list[i]+'.jpg'
		
		shrink_image(src).save(dstn)
		# shutil.copy(src,dstn)

	print ('testing done')
