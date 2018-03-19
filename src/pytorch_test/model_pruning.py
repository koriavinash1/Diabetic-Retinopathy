import os
import numpy as np
import time
import sys

import pandas as pd
from tqdm import tqdm

models_stats_df = pd.read_csv("../../All_models_Testing.csv")
models = np.squeeze(models_stats_df['model_used'].as_matrix())
models = np.unique(models)

tps = []
fps = []
total = []
tp_per = []
for model in models:
	print (model)
	model_info_df = models_stats_df[models_stats_df['model_used'] == model]
	_actual = np.squeeze(model_info_df['actual'].as_matrix())
	_pred = np.squeeze(model_info_df['predicted'].as_matrix())
	true_positive = np.sum(_actual==_pred)
	false_positive = len(_actual) - true_positive
	true_positive_percentage = float(true_positive)/(true_positive + false_positive)
	tps.append(true_positive)
	fps.append(false_positive)
	total.append(len(_actual))
	tp_per.append(true_positive_percentage)

sub = pd.DataFrame()
sub['models'] = models
sub['True_positives'] = tps
sub['False_positives'] = fps
sub['Total_examples'] = total
sub['True_positive_percentage'] = tp_per

print (sub)
sub.to_csv('../../model_pruning.csv', index=True)


def get_top_models(path, threshold = 0.95, total_networks = 8):
	data = pd.read_csv(path)
	# print (data)
	#expert_data = data[:total_networks]
	model1_data = data[:total_networks]
	# print (model1_data)
	max_tp_per_model1 = np.max(np.squeeze(model1_data['True_positive_percentage'].as_matrix()))
	# print (max_tp_per_expert)
	#max_tp_per_expert = np.max(np.squeeze(expert_data['True_positive_percentage'].as_matrix()))
	model1 = np.squeeze(model1_data[model1_data['True_positive_percentage'] >= threshold*max_tp_per_model1]['models'].as_matrix())
	#expert = np.squeeze(expert_data[expert_data['True_positive_percentage'] >= threshold*max_tp_per_expert]['models'].as_matrix())
	return model1

print ("#"*50)
m = get_top_models("../../model_pruning.csv", threshold = 0.95) 
print ("best models to consider based are :")
print (m)



