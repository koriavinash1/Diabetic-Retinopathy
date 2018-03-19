import pandas as pd
import numpy as np

csv_path = "../../Testing.csv"
nclasses = 5

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

