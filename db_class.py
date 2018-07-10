## Tensor Flow: Diabetes Classification (R dataset) ##
## 7/10/18 ##

# Import libs, set-up env:
from __future__ import absolute_import, division, print_function

import os
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

## Download diabetes dataset:

# Integer classifiers:
#0: Normal
#1: Chemical Diabetic
#2: Overt Diabetic

# Set input and output:
def parse_csv(line):
	pars = [[0.], [0.], [0.], [0.], [0.], [0]]
	parsed_line = tf.decode_csv(line, pars)
	preds = tf.reshape(parsed_line[:-1], shape=(5,))
	resp = tf.reshape(parsed_line[-1], shape=())
	return preds, resp

# Model Loss and Gradient functions for evaluation
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)


td = tf.data.TextLineDataset('/home/bubiea01/tensorflow/tutorials/diabetes_td.csv')
td = td.map(parse_csv)
td = td.shuffle(buffer_size=1000)
td = td.batch(8)

# Three hidden layer model:
model = tf.keras.Sequential([
	tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
	tf.keras.layers.Dense(10, activation='relu'),
	tf.keras.layers.Dense(3)
])

# Model optimizer:
optimizer = tf.train.GradientDescentOptimizer(learning_rate=5)

## TRAIN MODEL: ##

num_epochs = 501
for ep in range(num_epochs):
	epoch_loss_ave = tfe.metrics.Mean()
	epoch_acc = tfe.metrics.Accuracy()

	# Training loop - batches of 32 lines at a time:
	for p,r in td:
		grads = grad(model, p, r)
		optimizer.apply_gradients(zip(grads, model.variables),
									global_step=tf.train.get_or_create_global_step())

		epoch_loss_ave(loss(model, p, r))
		epoch_acc(tf.argmax(model(p), axis=1, output_type=tf.int32), r)

	if ep % 50 == 0:
		print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(ep, epoch_loss_ave.result(), epoch_acc.result()))



### EVALUATE MODEL: ###

# Set up Testing data:
fd = tf.data.TextLineDataset('/home/bubiea01/tensorflow/tutorials/diabetes.csv')
fd = fd.skip(1)
fd = fd.map(parse_csv)
fd = fd.shuffle(1000)
fd = fd.batch(8)

test_acc = tfe.metrics.Accuracy()

for (x,y) in fd:
	pred = tf.argmax(model(x), axis=1, output_type=tf.int32)
	test_acc(pred, y)

print("Test set accuracy: {:.3%}".format(test_acc.result()))