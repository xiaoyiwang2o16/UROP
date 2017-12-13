import numpy as np
import glob
import csv
import gpflow
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf
import math


class Brownian(gpflow.kernels.Kern):
    def __init__(self):
        gpflow.kernels.Kern.__init__(self, input_dim=344)
        self.variance = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        return tf.transpose(self.variance * tf.minimum(X, tf.transpose(X2)))

    def Kdiag(self, X):
        return self.variance * tf.reshape(X, (-1,))


class New_RBF(gpflow.kernels.Kern):
  def __init__(self):
      gpflow.kernels.Kern.__init__(self, input_dim=344)
      self.variance = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)

  def K(self, X, X2=None):
      if X2 is None:
          X2 = X
      gamma = tf.constant(-50.0)
      gamma = tf.cast(gamma, tf.float64)
      dist = tf.reduce_sum(tf.square(X), 1)
      dist = tf.reshape(dist, [-1,1])

      sq_dists = tf.add(tf.subtract(dist, tf.matmul(X, tf.transpose(X2))), tf.transpose(dist))
      my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
      #my_kernel = tf.exp(tf.multiply(self.variance, tf))
      return my_kernel

  def Kdiag(self, X):
      return tf.diag_part(self.K(X))



class New_RBF(gpflow.kernels.Kern):
  def __init__(self):
      gpflow.kernels.Kern.__init__(self, input_dim=344)
      self.variance = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)
      self.lengthscales = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)

  def K(self, X, X2=None):
      if X2 is None:
          X2 = X
      gamma = tf.constant(-0.5)
      gamma = tf.cast(gamma, tf.float64)
      dist = tf.reduce_sum(tf.square(X), 1)
      dist = tf.reshape(dist, [-1,1])


      sq_dists = tf.add(tf.subtract(dist, tf.matmul(X, tf.transpose(X2))), tf.transpose(dist))
      my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
      return self.variance**2*my_kernel**(1/(self.lengthscales**2))

  def Kdiag(self, X):
      gamma = tf.constant(-0.5)
      gamma = tf.cast(gamma, tf.float64)
      dist = tf.reduce_sum(tf.square(X), 1)
      dist = tf.reshape(dist, [-1,1])

      sq_dists = tf.add(tf.subtract(dist, tf.matmul(X, tf.transpose(X))), tf.transpose(dist))
      my_kernel = tf.exp((self.variance)*tf.multiply(gamma, tf.abs(sq_dists)))
      return tf.diag_part(my_kernel)

class New_RBF(gpflow.kernels.Kern):
  def __init__(self):
      gpflow.kernels.Kern.__init__(self, input_dim=1)
      self.variance = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)
      self.lengthscales = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)
      #self.variance = gpflow.param.Param(1.0)

  def K(self, X, X2=None):
      if X2 is None:
          X2 = X

      gamma = tf.cast(tf.constant(-0.5), tf.float64)
      #self.variance = tf.cast(self.variance, tf.float64)

      #dist = tf.reshape(tf.abs(tf.subtract(X, X2)), [-1, 1])
      dist = tf.abs(tf.subtract(X, X2))
      #dist = tf.multiply(self.variance, dist)
      my_kernel = tf.exp(tf.multiply(gamma, dist))
      return my_kernel*(1/self.variance)*(self.lengthscales)

  def Kdiag(self, X):
      return tf.diag_part(self.K(X))





class New_RBF(gpflow.kernels.Kern):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def __init__(self, input_dim, size):
        gpflow.kernels.Kern.__init__(self, input_dim=2)
        self.variance = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)
        lengthscale = np.ones(size)
        self.size= size
        self.lengthscales = gpflow.param.Param(lengthscale, transform=gpflow.transforms.positive)
        self.input_dim = input_dim

    def copy(self, ls, mul):
        result = []
        for i in range(len(mul)):
            result.append(np.tile(ls, (mul[i], 1)))
        result = np.array(result)
        return result


    def square_dist(self, X, X2):
        if self.input_dim > self.size:
            dim = self.input_dim/self.size
        else:
            dim = 1
        temp = tf.tile(self.lengthscales, [dim])
        X = X / temp
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1))  + tf.reshape(Xs, (1, -1))
            return dist


        X2 = X2 / temp
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        return dist

    def K(self, X, X2=None):
        return self.variance * tf.exp(-self.square_dist(X, X2) / 2)


    def Kdiag(self, X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))



# def gp():   
#   file = glob.glob('modified.csv')
#   with open(file[0], 'r') as csvfile:
#       reader = csv.reader(csvfile, delimiter=" ", quotechar= '|')
#       reader = list(reader)[1:]
#       X_train, X_test, Y_train, Y_test = [], [], [], []
#       for line in reader:
#           line = line[0].split(',')
#           if line[-1] == 'Train' or line[-1] == 'Val':
#               X_train.append(list(map(float, line[1:-6])))
#               Y_train.append([float(line[-5])])
#           elif line[-1] == 'Test':
#               X_test.append(list(map(float, line[1:-6])))
#               Y_test.append([float(line[-5])])

#       X_test = np.transpose(np.array(X_test).reshape(-1,len(X_test)))
#       Y_test = np.array(Y_test).reshape(len(Y_test), 1)
#       X_train = np.transpose(np.array(X_train).reshape(-1,len(X_train)))
#       Y_train = np.array(Y_train).reshape(len(Y_train), 1)

#       k = New_RBF()
#       #k = gpflow.kernels.RBF(input_dim=344)

        
#       m = gpflow.gpr.GPR(X_train, Y_train, kern=k)
#       print(m)

#       m.optimize()
#       #print(m)

#       mean, var = m.predict_f(X_train)
#       print(m)

#       plt.plot(Y_train)

#       plt.plot(mean)
#       plt.show()
#       # mean, var = m.predict_y(X_test)
#       # mean_absolute_error(mean, Y_test)
#       # plt.plot(Y_test)
#       # plt.plot(mean)
#       # plt.show()

# gp()



def test_points(n):
    X = np.random.random((100, 4))
    Y = []
    for row in X:
      Y.append(np.sum(np.sin(row)*6)+np.random.randn(1)*0.001)
    Y = np.array(Y)


    #k1 = gpflow.kernels.RBF(input_dim=4, ARD=True)
    k1 = New_RBF(input_dim=4, size=2)
    m = gpflow.gpr.GPR(X, Y, kern=k1)
    m.optimize()
    xx = np.random.random((500, 4))
    yy = []
    for row in xx:
      yy.append(np.sum(np.sin(row)*6)+np.random.randn(1)*0.001)
    yy = np.array(yy)


    mean, var = m.predict_y(xx)
    #plt.plot(X, Y, 'kx', mew=2)
    #line, = plt.plot(xx, mean, lw=2)
    #_ = plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color=line.get_color(), alpha=0.2)

    plt.plot(mean)
    print(m)
    plt.plot(yy)
    plt.show()
test_points(4)
