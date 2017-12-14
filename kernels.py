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
        #inherent from the kernels super class
        gpflow.kernels.Kern.__init__(self, input_dim=344)
        
        #create a variance variable
        self.variance = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)

    #implemented the kernel function
    def K(self, X, X2=None):
        #if there is no X2 input, initialize it to X
        if X2 is None:
            X2 = X
        #the brownian kernel just takes the smaller of the two values in X and X2
        return tf.transpose(self.variance * tf.minimum(X, tf.transpose(X2)))

    #returns the diagionals of the kernel function after it's been applied 
    def Kdiag(self, X):
        return self.variance * tf.reshape(X, (-1,))


    
    
class New_Kernel(gpflow.kernels.Kern):
    '''
    Kernel that simulates an RBF kernel without a lengthscales parameter
    '''
    def __init__(self):
        #initialize the kernel super class
        gpflow.kernels.Kern.__init__(self, input_dim=344)

        #create a variance parameter
        self.variance = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)

  #defined the kernel function
    def K(self, X, X2=None):
        
        #if we don't have an X2 input, make it X
        if X2 is None:
            X2 = X

        #create the constant 
        gamma = tf.constant(-50.0)
        gamma = tf.cast(gamma, tf.float64)

        #get the squared values of the X elements 
        dist = tf.reduce_sum(tf.square(X), 1)

        #reshape the array into a column tensor
        dist = tf.reshape(dist, [-1,1])


        #get squared values of X2 elements and subtract this number from dist
        sq_dists = tf.subtract(dist, tf.reshape(tf.reduce_sum(tf.square(X2), 1), [-1, 1]))

        #multiply by a constant and take the exponential of this
        my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

        #add the variance parameter
        return self.variance*my_kernel

    
  #return the diagonal values of the kernel
    def Kdiag(self, X):
        return tf.diag_part(self.K(X))


class Modified_RBF(gpflow.kernels.Kern):
  '''
    Kernel that simulates an RBF kernel with a lengthscales parameter
  '''
    def __init__(self):
        #inherit from the kernels super class
        gpflow.kernels.Kern.__init__(self, input_dim=1)

        #initialize a variance and a lengthscales parameter
        self.variance = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)
        self.lengthscales = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)

    #create the kernel function
    def K(self, X, X2=None):
        
        #if there is no X2 input, initialize it to X
        if X2 is None:
            X2 = X
        
        
        #initialize a constant to scale the distances
        gamma = tf.cast(tf.constant(-0.5), tf.float64)


        #get the absolute value of the difference between X and X2
        dist = tf.abs(tf.subtract(X, X2))
        
        #take the exponential of the lengthscales and the product of gamma and the distance between X and X2
        my_kernel = tf.exp((1/self.lengthscales)*tf.multiply(gamma, dist))
        
        #multiply by the variance parameter
        return my_kernel*self.variance

    
    #return the diagonal of the kernel function
    def Kdiag(self, X):
        return tf.diag_part(self.K(X))





class New_RBF(gpflow.kernels.Kern):
    """
    The radial basis function with a varied number of lengthscales
    """
    def __init__(self, input_dim, size):
        #inherit from super class
        gpflow.kernels.Kern.__init__(self, input_dim=2)
        self.input_dim = input_dim
        
        #size is the expected number of lengthscales
        self.size= size
        
        #initialize variance
        self.variance = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)
        
        #dictates the number of lengthscales we want
        lengthscale = np.ones(size)
        
        #initialize the correct number of lengthscales
        self.lengthscales = gpflow.param.Param(lengthscale, transform=gpflow.transforms.positive)
        

    #helper function that gets the squared distance between X and X2 
    def square_dist(self, X, X2):
        
        #make sure that the lengthscales match the number of dimensions in the X tensor
        dim = self.input_dim/self.size if self.input_dim > self.size else 1
        
        #tile the tensor to make the dimensions fit
        temp = tf.tile(self.lengthscales, [dim])
        
        #divide the X tensor with the lengthscales
        X = X / temp
        
        #get the sum of the square of all values in X
        X_square = tf.reduce_sum(tf.square(X), axis=1)
        
        #if there is no X2 value
        if X2 is None:
            
            #get the product of X and X and add the sum of the squares to it
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(X_square, (-1, 1))  + tf.reshape(X_square, (1, -1))
            return dist

    
        #otherwise if there is an X2, divide by the lengthscales
        X2 = X2 / temp
        
        #get the sum of the squares of all of the values in X2
        X2_square = tf.reduce_sum(tf.square(X2), axis=1)
        
        #get the product of X and X2
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        
        #add the sum of the squares of the values in X and X2
        dist += tf.reshape(X_square, (-1, 1)) + tf.reshape(X2_square, (1, -1))
        return dist

    #kernel function
    def K(self, X, X2=None):
        
        #return the variance times the exponential of the squared distance between X and X2
        return self.variance * tf.exp(-self.square_dist(X, X2) / 2)


    def Kdiag(self, X):
        
        #returns a tensor with just the variances along the 0th dimension of X
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


#function to predict engagement for the modified data
def gp():   
    
    #open the modified.csv file
    file = glob.glob('modified.csv')
    with open(file[0], 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=" ", quotechar= '|')
        reader = list(reader)[1:]
        X_train, X_test, Y_train, Y_test = [], [], [], []
        
        #go through the file
        for line in reader:
            line = line[0].split(',')
            #add training data to an array
            if line[-1] == 'Train':
                X_train.append(list(map(float, line[1:-6])))
                Y_train.append([float(line[-5])])
                
            #add testing/validation data to another array
            elif line[-1] == 'Test' or line[-1] == 'Val':
                X_test.append(list(map(float, line[1:-6])))
                Y_test.append([float(line[-5])])

        #transpose the arrays
        X_test = np.transpose(np.array(X_test).reshape(-1,len(X_test)))
        Y_test = np.array(Y_test).reshape(len(Y_test), 1)
        X_train = np.transpose(np.array(X_train).reshape(-1,len(X_train)))
        Y_train = np.array(Y_train).reshape(len(Y_train), 1)

        #initialize the kernel
        k = New_RBF(input_dim=344)
        #k = gpflow.kernels.RBF(input_dim=344)

        #create the gpflow model object
        m = gpflow.gpr.GPR(X_train, Y_train, kern=k)
        
        #optimize the model based on reducing squared error
        m.optimize()
        
        #predict y values based on the training data
        #mostly used as just a check that it's functioning properly
        mean, var = m.predict_f(X_train)
    
        #plot the predicted y values and the x values 
        plt.plot(Y_train)
        plt.plot(mean)
        plt.show()

        #predict results on the test data
        mean, var = m.predict_y(X_test)
    
        #plot the Y values
        plt.plot(Y_test)
        plt.plot(mean)
        plt.show()

# gp()


#helper function that tests the model on test points
def test_points():
    
    #generate random points with 4 dimensions in the x-coordinate
    X = np.random.random((100, 4))
    Y = []
    
    #for each row in x, take the sin of the row, sum it, and add random noise
    for row in X:
      Y.append(np.sum(np.sin(row)*6)+np.random.randn(1)*0.001)
   
    Y = np.array(Y)


    #k1 = gpflow.kernels.RBF(input_dim=4, ARD=True)
    
    #initialize kernel with 4 dimensions and 2 lengthscale parameters
    k1 = New_RBF(input_dim=4, size=2)
    
    #create the gpflow model object
    m = gpflow.gpr.GPR(X, Y, kern=k1)
    
    #optimize the model
    m.optimize()
    
    #create test data with 500 points
    xx = np.random.random((500, 4))
    
    yy = []
    
    #create y values through the same process as above
    for row in xx:
      yy.append(np.sum(np.sin(row)*6)+np.random.randn(1)*0.001)
    yy = np.array(yy)

    #predict the y values based on the given xx test points
    mean, var = m.predict_y(xx)
    
    #plot the predictions and the correct y values
    plt.plot(mean)
    plt.plot(yy)
    plt.show()
    
    
test_points()
