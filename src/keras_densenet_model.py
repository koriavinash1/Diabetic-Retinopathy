from keras.applications.densenet import DenseNet201
import tensorflow as tf
import numpy as np

model = DenseNet201(include_top=False, weights='imagenet')
model.summary()

