# import gpflow
# import numpy as np
# from matplotlib import pyplot as plt
# plt.style.use('ggplot')

# N = 12
# X = np.random.rand(N,1)
# Y = np.sin(12*X)+0.66*np.cos(25*X)+np.random.randn(N,1)*0.1+3
# plt.plot(X, Y, 'kx', mew=2)
# plt.show()

# k = gpflow.kernels.Matern52(1, lengthscales=0.3)
# m = gpflow.gpr.GPR(X, Y, kern=k)
# m.likelihood.variance = 0.01


import gpflow
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf
#%matplotlib inline


N = 100
X = np.random.rand(N,1)
Y = np.sin(12*X)+0.66*np.cos(25*X)+np.random.randn(N,1)*0.1+3
#plt.plot(X, Y, 'kx', mew=2)
#plt.show()


k = gpflow.kernels.Matern52(1, lengthscales=0.3)
m = gpflow.gpr.GPR(X, Y, kern=k)
m.likelihood.variance = 0.01

# k = gpflow.kernels.Matern52(1, lengthscales=0.3)
# meanf = gpflow.mean_functions.Linear(1,0)
# m = gpflow.gpr.GPR(X, Y, k, meanf)
# m.likelihood.variance = 0.01

def plot(m):
    xx = np.linspace(-0.1, 1.1, 100)[:,None]
    mean, var = m.predict_y(xx)
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)
    plt.xlim(-0.1, 1.1)
    plt.show()
#plot(m)
m.optimize()
plot(m)








'''
user_id, 
remove a and c columns
from d to MI, nromalize for entire column, 
	z-score ... subtract mean and divide by SD
from MJ to MN: go from zero to 1
use mean absolute error and interclass correlaion for error
rbf kernel
use default initial parameters, 
'''
