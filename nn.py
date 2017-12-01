import pdb
import numpy as np

from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.datasets import mnist
from keras import backend as K
from matplotlib import pyplot as plt

# # Read the simple 2D dataset files
# def get_data_set(name):
#     try:
#         data = np.loadtxt(name, skiprows=0, delimiter = ' ')
#     except:
#         return None, None, None
#     np.random.shuffle(data)             # shuffle the data
#     # The data uses ROW vectors for a data point, that's what Keras assumes.
#     _, d = data.shape
#     X = data[:,0:d-1]
#     Y = data[:,d-1:d]
#     y = Y.T[0]
#     classes = set(y)
#     if classes == set([-1.0, 1.0]):
#         print('Convert from -1,1 to 0,1')
#         y = 0.5*(y+1)
#     print('Loading X', X.shape, 'y', y.shape, 'classes', set(y))
#     return X, y, len(classes)

# class LossHistory(Callback):
#     def on_train_begin(self, logs={}):
#         self.keys = ['loss', 'acc', 'val_loss', 'val_acc']
#         self.values = {}
#         for k in self.keys:
#             self.values['batch_'+k] = []
#             self.values['epoch_'+k] = []

#     def on_batch_end(self, batch, logs={}):
#         for k in self.keys:
#             bk = 'batch_'+k
#             if k in logs:
#                 self.values[bk].append(logs[k])

#     def on_epoch_end(self, epoch, logs={}):
#         for k in self.keys:
#             ek = 'epoch_'+k
#             if k in logs:
#                 self.values[ek].append(logs[k])

#     def plot(self, keys):
#         for key in keys:
#             plt.plot(np.arange(len(self.values[key])), np.array(self.values[key]), label=key)
#         plt.legend()

# def run_keras(X_train, y_train, X_val, y_val, X_test, y_test, layers, epochs, split=0):
#     # Model specification
#     model = Sequential()
#     for layer in layers:
#         model.add(layer)
#     # Define the optimization
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
#     N = X_train.shape[0]
#     # Pick batch size
#     batch = 32 if N > 1000 else 1     # batch size
#     history = LossHistory()
#     # Fit the model
#     if X_val is None:
#         model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_split=split,
#                   callbacks=[history])
#     else:
#         model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_val, y_val),
#                   callbacks=[history])
#     # Evaluate the model on test data, if any
#     if X_test is not None:
#         objective_score = model.evaluate(X_test, y_test, batch_size=batch)
#         # objective_score is a tuple containing the loss as well as the accuracy
#         print ("\nLoss on test set:"  + str(objective_score[0]) + " Accuracy on test set: " + str(objective_score[1]))
#     return model, history

# def dataset_paths(data_name):
#     return ["data/data"+data_name+"_"+suffix+".csv" for suffix in ("train", "validate", "test")]

# # The name is a string such as "1" or "Xor"
# def run_keras_2d(data_name, layers, epochs, display=True, split=0.25):
#     print('Keras FC: dataset=', data_name)
#     (train_dataset, val_dataset, test_dataset) = dataset_paths(data_name)
#     # Load the datasets
#     X_train, y, num_classes = get_data_set(train_dataset)
#     X_val, y2, _ = get_data_set(val_dataset)
#     X_test, y3, _ = get_data_set(test_dataset)
#     # Categorize the labels
#     y_train = np_utils.to_categorical(y, num_classes) # one-hot
#     if X_test is not None:
#         y_val = np_utils.to_categorical(y2, num_classes) # one-hot
#         y_test = np_utils.to_categorical(y3, num_classes) # one-hot
#     else:
#         y_val = y_test = None
#     # Run the model
#     model, history = run_keras(X_train, y_train, X_val, y_val, X_test, y_test, layers, epochs, split=split)
#     if display:
#         # plot classifier landscape on training data
#         plot_heat(X_train, y, model)
#         plt.title('Training data')
#         plt.show()
#         if X_test is not None:
#             # plot classifier landscape on testing data
#             plot_heat(X_test, y3, model)
#             plt.title('Testing data')
#             plt.show()
#         # Plot epoch accuracy
#         history.plot(['epoch_acc', 'epoch_val_acc'])
#         plt.xlabel('epoch')
#         plt.ylabel('accuracy')
#         plt.title('Epoch val_acc and acc')
#         plt.show()
#     # To cleanup, but then we can't use the model any more.
#     # K.clear_session()
#     return model, history

# def get_MNIST_data(shift=0):
#     (X_train, y1), (X_test, y2) = mnist.load_data()
#     if shift:
#         size = 28+shift
#         X_train = shifted(X_train, shift)
#         X_test = shifted(X_test, shift)
#     #return (X_train, y1), (X_test, y2)
#     return (np.divide(X_train, 255.0), y1), (np.divide(X_test, 255.0), y2)

# def shifted(X, shift):
#     n = X.shape[0]
#     m = X.shape[1]
#     size = m + shift
#     X_sh = np.zeros((n, size, size))
#     plt.ion()
#     for i in range(n):
#         sh1 = np.random.randint(shift)
#         sh2 = np.random.randint(shift)
#         X_sh[i, sh1:sh1+m, sh2:sh2+m] = X[i, :, :]
#         # If you want to see the shifts, uncomment
#         # plt.figure(1); plt.imshow(X[i])
#         # plt.figure(2); plt.imshow(X_sh[i])
#         # plt.show()
#         # input('Go?')
#     return X_sh

# def run_keras_fc_mnist(train, test, layers, epochs, split=0.1):
#     (X_train, y1), (X_test, y2) = train, test
#     # Flatten the images
#     m = X_train.shape[1]
#     X_train = X_train.reshape((X_train.shape[0], m*m))
#     X_test = X_test.reshape((X_test.shape[0], m*m))
#     # Categorize the labels
#     num_classes = 10
#     y_train = np_utils.to_categorical(y1, num_classes)
#     y_test = np_utils.to_categorical(y2, num_classes)
#     # Train, use split for validation
#     return run_keras(X_train, y_train, None, None, X_test, y_test, layers, epochs, split=split)

# def run_keras_cnn_mnist(train, test, layers, epochs, split=0.1):
#     # Load the dataset
#     (X_train, y1), (X_test, y2) = train, test
#     # Add a final dimension indicating the number of channels (only 1 here)
#     m = X_train.shape[1]
#     X_train = X_train.reshape((-1, m, m, 1))
#     X_test = X_test.reshape((-1, m, m, 1))
#     # Categorize the labels
#     num_classes = 10
#     y_train = np_utils.to_categorical(y1, num_classes)
#     y_test = np_utils.to_categorical(y2, num_classes)
#     # Train, use split for validation
#     return run_keras(X_train, y_train, None, None, X_test, y_test, layers, epochs, split=split)

# # Plotting functions

# def plot_heat(X, y, model, res = 200):
#     eps = .1
#     xmin = np.min(X[:,0]) - eps; xmax = np.max(X[:,0]) + eps
#     ymin = np.min(X[:,1]) - eps; ymax = np.max(X[:,1]) + eps
#     ax = tidyPlot(xmin, xmax, ymin, ymax, xlabel = 'x', ylabel = 'y')
#     xl = np.linspace(xmin, xmax, res)
#     yl = np.linspace(ymin, ymax, res)
#     xx, yy = np.meshgrid(xl, yl, sparse=False)
#     zz = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
#     im = ax.imshow(np.flipud(zz.reshape((res,res))), interpolation = 'none',
#                    extent = [xmin, xmax, ymin, ymax],
#                    cmap = 'viridis')
#     plt.colorbar(im)
#     colors = [['r', 'g', 'b'][int(l)] for l in y]
#     ax.scatter(X[:,0], X[:,1], c = colors, marker = 'o', s=80,
#                              edgecolors = 'none')

# def tidyPlot(xmin, xmax, ymin, ymax, center = False, title = None,
#                  xlabel = None, ylabel = None):
#     plt.figure(facecolor="white")
#     ax = plt.subplot()
#     if center:
#         ax.spines['left'].set_position('zero')
#         ax.spines['right'].set_color('none')
#         ax.spines['bottom'].set_position('zero')
#         ax.spines['top'].set_color('none')
#         ax.spines['left'].set_smart_bounds(True)
#         ax.spines['bottom'].set_smart_bounds(True)
#         ax.xaxis.set_ticks_position('bottom')
#         ax.yaxis.set_ticks_position('left')
#     else:
#         ax.spines["top"].set_visible(False)    
#         ax.spines["right"].set_visible(False)    
#         ax.get_xaxis().tick_bottom()  
#         ax.get_yaxis().tick_left()
#     eps = .05
#     plt.xlim(xmin-eps, xmax+eps)
#     plt.ylim(ymin-eps, ymax+eps)
#     if title: ax.set_title(title)
#     if xlabel: ax.set_xlabel(xlabel)
#     if ylabel: ax.set_ylabel(ylabel)
#     return ax


# # # ##XOR no hidden layers
# # run_keras_2d('Xor', [Dense(input_shape=(2,), units=2, activation='softmax')], 100, split=0)
# # ###


# # ##XOR separation with relu units
# # #2 relu units and softmax should be able to do it 
# # run_keras_2d('Xor', [Dense(input_shape=(2,), units=2, activation='relu'), Dense(units=2, activation='softmax')], 100, split=0)
# # ###


# # ##XOR separation  with sigmoid units
# # #2 relu units and softmax should be able to do it 
# # run_keras_2d('Xor', [Dense(input_shape=(2,), units=2, activation='sigmoid'), Dense(units=2, activation='softmax')], 100, split=0)
# # ###


# # ##XOR separation 
# # #10 relu units and softmax should be able to do it 
# # run_keras_2d('Xor', [Dense(input_shape=(2,), units=5, activation='relu'), Dense(units=2, activation='softmax')], 1000, split=0)
# # ###

# # ##XOR separation 
# # #2 layers with 5 hidden units each == same number of hidden units as above but split into two 
# # run_keras_2d('Xor', [Dense(input_shape=(2,), units=2, activation='relu'), Dense(units=3, activation='relu'), Dense(units=2, activation='softmax')], 2000, split=0)
# # ###

# # ##2 D Dataset
# # #data 1
# # run_keras_2d('1', [Dense(units=2, input_shape=(2,), activation='softmax')], 10, split=0.25)


# ##2 D Dataset
# #data 2
# run_keras_2d('2', [Dense(input_shape=(2,), units=100, activation='relu'), Dense(units=100, activation='relu'), Dense(units=100, activation='relu'), Dense(units=100, activation='relu'), Dense(units=2, activation='softmax')], 10, split=0.25)

# # ##2 D Dataset
# # #data 3
# # run_keras_2d('3', [Dense(input_shape=(2,), units=100, activation='relu'), Dense(units=100, activation='relu'), Dense(units=2, activation='softmax')], 10, split=0.25)

# # ##2 D Dataset
# # #data 4
# # run_keras_2d('4', [Dense(input_shape=(2,), units=10, activation='relu'), Dense(units=2, activation='softmax')], 10, split=0.25)

# # #all of them 2d data sets
# # for i in range(2, 5):
# #     run_keras_2d(str(i), [Dense(input_shape=(2,), units=10, activation='relu'), Dense(units=100, activation='relu'), Dense(units=2, activation='softmax')], 10, split=0.25)

# # ##three-class data set
# # run_keras_2d('3class', [Dense(units=3,input_dim=2, activation='softmax')], 10, split=0.5)


# # ##three-class data set
# # run_keras_2d('3class', [Dense(input_dim=2, units=1000, activation='relu'), Dense(units=3, activation='softmax')], 10, split=0.5)


# # #with no hidden layers
# # train, test = get_MNIST_data()
# # run_keras_fc_mnist(train, test, [Dense(units=10, input_dim=784, activation='softmax')], 1, split=0.1)


# # #with only a single layer
# # train, test = get_MNIST_data()
# # run_keras_fc_mnist(train, test, [Dense(input_dim=784, units=1024, activation='relu'), Dense(units=10, activation='softmax')], 1, split=0.1)

# # #with two layers
# # train, test = get_MNIST_data()
# # run_keras_fc_mnist(train, test, [Dense(input_dim=784, units=512, activation='relu'), Dense(units=256, activation='relu'), Dense(units=10, activation='softmax')], 1, split=0.1)


# # #Convolution neural network
# # train, test = get_MNIST_data()
# # run_keras_cnn_mnist(train, test, [Convolution2D(input_shape=(28, 28, 1), filters=32, kernel_size=(3, 3), activation='relu'), MaxPooling2D(pool_size=(2, 2)), 
# #     Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'),
# #     MaxPooling2D(pool_size=(2, 2)),
# #     Flatten(),
# #     Dense(units=128, activation='relu'),
# #     Dropout(rate=0.5), 
# #     Dense(units=10, activation='softmax')], 1, split=0.1)


# # ##compare fc or cnn
# # train_20, test_20 = get_MNIST_data(shift=20)
# # # run_keras_cnn_mnist(train_20, test_20, [Convolution2D(input_shape=(48, 48, 1), filters=32, kernel_size=(3, 3), activation='relu'), MaxPooling2D(pool_size=(2, 2)), 
# # #     Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'),
# # #     MaxPooling2D(pool_size=(2, 2)),
# # #     Flatten(),
# # #     Dense(units=128, activation='relu'),
# # #     Dropout(rate=0.5), 
# # #     Dense(units=10, activation='softmax')], 1, split=0.1)
# # train_20, test_20 = get_MNIST_data(shift=20)
# # run_keras_fc_mnist(train_20, test_20, [Dense(input_dim=2304, units=512, activation='relu'), Dense(units=256, activation='relu'), Dense(units=10, activation='softmax')], 1, split=0.1)
import numpy as np
import glob
import csv
import gpflow
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf
import math
def gp():   
	file = glob.glob('modified.csv')
	with open(file[0], 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=" ", quotechar= '|')
		reader = list(reader)[1:]
		X_train, X_test, Y_train, Y_test = [], [], [], []
		for line in reader:
			line = line[0].split(',')
			if line[-1] == 'Train' or line[-1] == 'Val':
				X_train.append(list(map(float, line[1:-6])))
				Y_train.append([float(line[-5])])
			elif line[-1] == 'Test':
				X_test.append(list(map(float, line[1:-6])))
				Y_test.append([float(line[-5])])

		X_test = np.transpose(np.array(X_test).reshape(-1,len(X_test)))
		Y_test = np.array(Y_test).reshape(len(Y_test), 1)
		X_train = np.transpose(np.array(X_train).reshape(-1,len(X_train)))
		Y_train = np.array(Y_train).reshape(len(Y_train), 1)

		net_act = Sequential()
		net_act.add(Dense(input_shape=(344,), units=1000, activation='relu'))
		for i in range(10):
			net_act.add(Dense(units=1000, activation='relu'))
		net_act.add(Dense(units=1, activation='linear'))
		net_act.compile(loss="mean_squared_error", optimizer=Adam(), metrics=['accuracy'])

		for i in range(len(X_train)):
			inp = X_train[i]
			inp = inp.reshape(1, -1)
			out = Y_train[i]
			out = out.reshape(1, 1)
			net_act.fit(inp, out, epochs=1, verbose=False)
			print(i)

		for i in range(len(X_test)):
			inp = X_test[i]
			inp = inp.reshape(1, -1)
			out = Y_test[i]
			out = out.reshape(1, 1)

			ans = net_act.predict(inp)
			print(ans)
gp()
