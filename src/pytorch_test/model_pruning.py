import os
import numpy as np
import time
import sys

import pandas as pd
from tqdm import tqdm

models_stats_df = pd.read_csv("../../Varghese_Testing_PerModel_PerImage.csv")
models = np.squeeze(models_stats_df['model_used'].as_matrix())
models = np.unique(models)
print (models)

tps = []
fps = []
total = []
tp_per = []
f1_s = []

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
	f1_s.append(float(2.0*true_positive)/(2*true_positive + false_positive))

sub = pd.DataFrame()
sub['models'] = models
sub['True_positives'] = tps
sub['False_positives'] = fps
sub['Total_examples'] = total
sub['True_positive_percentage'] = tp_per
sub['F1_Score'] = f1_s

print (sub)
sub.to_csv('../../model_pruning.csv', index=True)


def get_top_models(path, threshold = 0.98):
	data = pd.read_csv(path)
	# print (data)
	expert_data = data[[str.__contains__('expert') for str in data['models'].as_matrix()]]
	model1_data = data[[not str.__contains__('expert') for str in data['models'].as_matrix()]]
	# print (model1_data)
	max_tp_per_model1 = np.max(np.squeeze(model1_data['F1_Score'].as_matrix()))
	max_tp_per_expert = np.max(np.squeeze(expert_data['F1_Score'].as_matrix()))

	model1 = np.squeeze(model1_data[model1_data['F1_Score'] >= threshold*max_tp_per_model1]['models'].as_matrix())
	expert = np.squeeze(expert_data[expert_data['F1_Score'] >= threshold*max_tp_per_expert]['models'].as_matrix())
	return model1, expert

print ("#"*50)
m, e = get_top_models("../../model_pruning.csv", threshold = 0.95) 
print ("best models to consider based are :")
print (m)

print ("best expert models to consider based are :")
print (e)


