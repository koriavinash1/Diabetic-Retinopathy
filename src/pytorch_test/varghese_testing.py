import os
import torch
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import numpy as np
import pandas as pd
IMAGENET_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transformListIMAGENET = []
transformListIMAGENET.append(transforms.Resize(256))
transformListIMAGENET.append(transforms.TenCrop(224))
transformListIMAGENET.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
transformListIMAGENET.append(transforms.Lambda(lambda crops: torch.stack([IMAGENET_normalize(crop) for crop in crops])))
transformIMAGENET = transforms.Compose(transformListIMAGENET)


IRID_normalize = transforms.Normalize([0.511742964836, 0.243537961753, 0.0797484182405], [0.223165616204, 0.118469339976, 0.0464971614141])
transformListIRID = []
transformListIRID.append(transforms.Resize(256))
transformListIRID.append(transforms.TenCrop(224))
transformListIRID.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
transformListIRID.append(transforms.Lambda(lambda crops: torch.stack([IRID_normalize(crop) for crop in crops])))
transformIRID=transforms.Compose(transformListIRID)


path_to_test = '/home/brats/drbackup/DR2/processed_data/test'
classes= os.listdir(path_to_test)

class_of_interest =4
path_containing_primary_models = '/home/brats/drbackup/DR2/v_model/primary_a'
path_containing_expert_models = '/home/brats/drbackup/DR2/v_model/expert_a'

# files = os.listdir(path_to_test)

trained_primary_models = os.listdir(path_containing_primary_models) 
trained_expert_models  = os.listdir(path_containing_expert_models)

cntr =0
total_number_of_files=0

final_model_prediction= []


_outGT=[]
_outPREDs=[]
_image_paths = []
_mused = []

for c in classes:
	cntr=0
	print ('class',c)
	if c =='class4':
		class_of_interest=4
	if c =='class3':
		class_of_interest=3
	if c =='class2':
		class_of_interest=2
	if c =='class1':
		class_of_interest=1
	if c =='class0':
		class_of_interest=0	
	files = os.listdir(path_to_test+'/'+c)

	for f in files:

		if 'jpg' or 'png' in f:

			test_data = Image.open(path_to_test+'/'+c+'/'+f).convert('RGB')
			image_source = path_to_test+'/'+c+'/'+f
			# print ('currently testing',f)
	 
			model_prediction =[]

			for model in trained_primary_models:

				# print ('testing with model===>',model)
				if 'IMAGENET' in model:
					transformed_image = transformIMAGENET(test_data)   ### currently in 10,3,224,224
				if 'IRID' in model:
					print('loading IRID stats')
					transformed_image = transformIRID(test_data)
				

				model_to_test =torch.load(path_containing_primary_models+'/'+model)
				model_source  =path_containing_primary_models+'/'+model
				# model_to_test.eval()

				outs = model_to_test(Variable(transformed_image,volatile=True))
				del model_to_test
				del transformed_image

				_,class_associated_to_each_Crop = torch.max(outs,1)

				del _
				del outs

				class_associated_to_each_Crop = class_associated_to_each_Crop.data.cpu().numpy()
				unique,counts = np.unique(class_associated_to_each_Crop,return_counts=True)

				del class_associated_to_each_Crop

				image_pred = unique[np.argmax(counts)] ### one max done
				del unique
				del counts

				model_prediction.append(image_pred)

				### logging as avinash said so

				if image_pred !=3:
					_outGT.append(class_of_interest)
					_outPREDs.append(image_pred)
					_image_paths.append(image_source)
					_mused.append(model_source)

			## technically model prediction should contain as many predictions as the number of primary models

			## time to do second max,
			model_prediction= np.array(model_prediction)
			unique,counts = np.unique(model_prediction,return_counts=True)
			primay_prediction = unique[np.argmax(counts)]
			del unique
			del counts

			del model_prediction


			if primay_prediction ==3:   ## if statisfied, ensemble has assigned the class 3 (S-NDPR), use expert models
				expert_prediction=[]
				for e_model in trained_expert_models:

					if 'IMAGENET' in e_model:
						transformed_image = transformIMAGENET(test_data)   ### currently in 10,3,224,224

					if 'IRID' in e_model:
						transformed_image = transformIRID(test_data)	

					expert_model_to_test = torch.load(path_containing_expert_models+'/'+e_model)
					expert_source        = path_containing_expert_models+'/'+e_model
					# expert_model_to_test.eval()
					outs = expert_model_to_test(Variable(transformed_image,volatile=True))
					del expert_model_to_test
					del transformed_image

					_,class_associated_to_each_Crop = torch.max(outs,1)

					del _
					del outs

					class_associated_to_each_Crop = class_associated_to_each_Crop.data.cpu().numpy()
					unique,counts = np.unique(class_associated_to_each_Crop,return_counts=True)
					del class_associated_to_each_Crop

					if unique[np.argmax(counts)] ==0:    ### first max over
						image_pred =3
					else :
						image_pred =4

					del unique
					del counts

					expert_prediction.append(image_pred)
					_outGT.append(class_of_interest)
					_outPREDs.append(image_pred)
					_image_paths.append(image_source)
					_mused.append(expert_source)

				expert_prediction = np.array(expert_prediction)
				unique,counts = np.unique(expert_prediction,return_counts=True)

				final_prediction = unique[np.argmax(counts)]
				del unique
				del counts
				del expert_prediction
			

			else: 
				final_prediction = primay_prediction


			if final_prediction == class_of_interest:
				cntr= cntr+1

			# del transformed_image


	print ("counter: ", cntr)


sub = pd.DataFrame()
sub['path'] = _image_paths
sub['actual'] = _outGT
sub['predicted'] = _outPREDs
sub['model_used'] = _mused
# sub.to_csv('../../Varghese_Testing_PerModel_PerImage.csv', index=True)










