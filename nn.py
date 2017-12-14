import pdb


from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.datasets import mnist
from keras import backend as K
from matplotlib import pyplot as plt

import numpy as np
import glob
import csv
import gpflow
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf
import math


def nn():   
	#get the files
	file = glob.glob('modified.csv')
	with open(file[0], 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=" ", quotechar= '|')
		reader = list(reader)[1:]
		X_train, X_test, Y_train, Y_test = [], [], [], []
		
		#get the training, test, and validation data
		for line in reader:
			line = line[0].split(',')
			if line[-1] == 'Train' or line[-1] == 'Val':
				X_train.append(list(map(float, line[1:-6])))
				Y_train.append([float(line[-5])])
			elif line[-1] == 'Test':
				X_test.append(list(map(float, line[1:-6])))
				Y_test.append([float(line[-5])])

		#transpose the arrays
		X_test = np.transpose(np.array(X_test).reshape(-1,len(X_test)))
		Y_test = np.array(Y_test).reshape(len(Y_test), 1)
		X_train = np.transpose(np.array(X_train).reshape(-1,len(X_train)))
		Y_train = np.array(Y_train).reshape(len(Y_train), 1)
	
		#create a neural network with 10 layers and 1000 units in each layer
		net_act = Sequential()
		net_act.add(Dense(input_shape=(344,), units=1000, activation='relu'))
		for i in range(10):
			net_act.add(Dense(units=1000, activation='relu'))
		net_act.add(Dense(units=1, activation='linear'))
		net_act.compile(loss="mean_squared_error", optimizer=Adam(), metrics=['accuracy'])

		#train the model on each data point
		for i in range(len(X_train)):
			inp = X_train[i]
			inp = inp.reshape(1, -1)
			out = Y_train[i]
			out = out.reshape(1, 1)
			net_act.fit(inp, out, epochs=1, verbose=False)
			print(i)
		
		#try to predict on each data point
		for i in range(len(X_test)):
			inp = X_test[i]
			inp = inp.reshape(1, -1)
			out = Y_test[i]
			out = out.reshape(1, 1)

			ans = net_act.predict(inp)
			print(ans)
			
			
nn()
