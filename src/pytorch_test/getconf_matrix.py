import pandas as pd
import numpy as np

csv_path = "../../Testing.csv"
nclasses = 5

data = pd.read_csv(csv_path)
actual = data['actual'].as_matrix()
pred = data['predicted'].as_matrix()

conf_matrix = []

for i in range(nclasses):
	temp = []
	index = np.where(actual == i)[0]
	for j in range(nclasses):
		temp.append(np.sum(pred[index] == j))
	conf_matrix.append(temp)

print ("confusion matrix ----")
for c in conf_matrix:
	print (c)
