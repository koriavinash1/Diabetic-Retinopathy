import pandas as pd
import numpy as np

dr_csv_path = "./DR_submission.csv"
dme_csv_path = "./DME_submission.csv"
nclasses = 5

DR_data = pd.read_csv(dr_csv_path)
DR_img_ids = DR_data['ID'].as_matrix()
DR_pred = np.squeeze(DR_data['DR'].as_matrix())

DME_data = pd.read_csv(dme_csv_path)
DME_img_ids = DME_data['Image No'].as_matrix()
DME_grades = DME_data['Risk of DME'].as_matrix()

DR_grades = []
for img_id in DME_img_ids:
	index = np.where(DR_img_ids == img_id)[0]
	print(DR_img_ids[index], img_id)
	DR_grades.append(DR_pred[index][0])

sub = pd.DataFrame()
sub['Image No'] = DME_img_ids
sub['DR Grade'] = DR_grades
sub['Risk of DME'] = DME_grades

sub.to_csv('./final_submission.csv', index=False)









"""
data = pd.read_csv(csv_path)
image_paths = data['path'].as_matrix()
pred = data['predicted'].as_matrix()

ids = []
pred_DR = []
pred_DME = []
 
for i in range(len(image_paths)):
	path = image_paths[i]

	id, ext = path.split("/").pop().split(".")
	if ext == "jpg',)":
		ids.append(id)
		pred_DR.append(pred[i])
		pred_DME.append(0)


print ("Submission file ----")

sub = pd.DataFrame()
sub['ID'] = ids
sub['DR'] = pred_DR
sub['DME'] = pred_DME

sub.to_csv('../../submission.csv', index=False)
"""

