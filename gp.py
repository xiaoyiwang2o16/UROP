import numpy as np
import glob
import csv
import gpflow
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf

def _icc(y_hat, y_lab, cas=3, typ=1):

	def fun(y_hat,y_lab):
		y_hat = y_hat[None,:]
		y_lab = y_lab[None,:]

		Y = np.array((y_lab, y_hat))
		# number of targets
		n = Y.shape[2]

		# mean per target
		mpt = np.mean(Y, 0)

		# print mpt.eval()
		mpr = np.mean(Y, 2)

		# print mpr.eval()
		tm = np.mean(mpt, 1)

		# within target sum sqrs
		WSS = np.sum((Y[0]-mpt)**2 + (Y[1]-mpt)**2, 1)

		# within mean sqrs
		WMS = WSS/n

		# between rater sum sqrs
		RSS = np.sum((mpr - tm)**2, 0) * n

		# between rater mean sqrs
		RMS = RSS

		# between target sum sqrs
		TM = np.tile(tm, (y_hat.shape[1], 1)).T
		BSS = np.sum((mpt - TM)**2, 1) * 2

		# between targets mean squares
		BMS = BSS / (n - 1)

		# residual sum of squares
		ESS = WSS - RSS

		# residual mean sqrs
		EMS = ESS / (n - 1)

		if cas == 1:
			if typ == 1:
				res = (BMS - WMS) / (BMS + WMS)
			if typ == 2:
				res = (BMS - WMS) / BMS
		if cas == 2:
			if typ == 1:
				res = (BMS - EMS) / (BMS + EMS + 2 * (RMS - EMS) / n)
			if typ == 2:
				res = (BMS - EMS) / (BMS + (RMS - EMS) / n)
		if cas == 3:
			if typ == 1:
				res = (BMS - EMS) / (BMS + EMS)
			if typ == 2:
				res = (BMS - EMS) / BMS

		res = res[0]

		if np.isnan(res) or np.isinf(res):
			return 0
		else:
			return res
	return _process(y_hat, y_lab, fun)



class newRBF(gpflow.model.Model):
    def __init__(self, X, Y):
	# call the parent constructor
        gpflow.model.Model.__init__(self) 
	
	# X is a numpy array of inputs
        self.X = X.copy() 
	
	# Y is a representation of the labels
        self.Y = Y.copy() 

        self.num_data, self.input_dim = X.shape
        self.num_classes = Y.shape[1]

        #make some parameters
        self.W = gpflow.param.Param(np.random.randn(self.input_dim, self.num_classes))
        self.b = gpflow.param.Param(np.random.randn(self.num_classes))



    def build_likelihood(self): 
	
	# Param variables are used as tensors
        p = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b) 
		
	#return sum of Y*log(p)
        return tf.reduce_sum(tf.log(p) * self.Y) 




def mean_absolute_error(pred, labels):
	print(sum(abs(pred-labels))/len(labels))

def gp():	

	#open the file
	file = glob.glob('modified.csv')
	with open(file[0], 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=" ", quotechar= '|')
		reader = list(reader)[1:]
		X_train = []
		Y_train = []
		X_test = []
		Y_test = []
		
		#add the training, validation, and testing data
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

		
		#practicing with different kernels
		k3 = gpflow.kernels.RBF(input_dim=344, ARD=False)
		k1 = gpflow.kernels.Linear(input_dim=344)
		k6 = gpflow.kernels.Linear(input_dim=344)
		#k2 = gpflow.kernels.Constant(input_dim=344)
		#k4 = gpflow.kernels.Polynomial(input_dim=344)
		k5 = gpflow.kernels.White(input_dim=344)
		
		#create the GPR model object
		m = gpflow.gpr.GPR(X_train, Y_train, kern=k1+k6)




		#optimize the model
		m.optimize()


		#predict based on the training data
		mean, var = m.predict_f(X_train)

		#plot the Y and the predicted Y
		plt.plot(Y_train)

		plt.plot(mean)

		plt.show()
		
		
		#predict on the test data
		mean, var = m.predict_y(X_test)
		
		#plot the expected Y and the real Y
		
		plt.plot(Y_test)
		plt.plot(mean)
		plt.show()

		#get the mean absolute error on the predicted values
		mean_absolute_error(mean, Y_test)


gp()
