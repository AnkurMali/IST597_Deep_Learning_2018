import os  
import tensorflow as tf
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  

'''
IST 597: Foundations of Deep Learning
Problem 1: Univariate Regression solution

@author - Alexander G. Ororbia II and Ankur Mali

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
alpha = 0.01 # step size coefficient
eps = 0.00001 # controls convergence criterion
n_epoch = 300 # number of epochs (full passes through the dataset)

# begin simulation


path = os.getcwd() + '/data/prob1.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 

# display some information about the dataset itself here
print(data.shape)
# WRITEME: write your code here to print out information/statistics about the data-set "data" using Pandas (consult the Pandas documentation to learn how)
print(data.describe())
# WRITEME: write your code here to create a simple scatterplot of the dataset itself and print/save to disk the result
plt.scatter(data.iloc[:,0], data.iloc[:,1], c= 'red', marker = "+")
plt.savefig(os.getcwd() + '/Pb_1_Scattered_plot')

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
X = np.float32(X)
y = np.float32(y)
# make results reproducible
seed = 12345
np.random.seed(seed)
tf.set_random_seed(seed)

#TODO convert np array to tensor objects
X_in = tf.convert_to_tensor(X, dtype = tf.float32)
y_in = tf.convert_to_tensor(y, dtype = tf.float32)

 

#TODO create an placeholder variable for X(input) and Y(output)
X_p = tf.placeholder(tf.float32, shape = (97, 1))
y_p = tf.placeholder(tf.float32, shape = (97, 1))
w_p = tf.placeholder(tf.float32 , shape = (None,None))
b_p = tf.placeholder(tf.float32 , shape = (None))
dw_p = tf.placeholder(tf.float32 , shape=(None,None))
db_p = tf.placeholder(tf.float32 , shape =(None))

# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1,X.shape[1]))
print(w.shape)
b = np.array([0])
print(b.shape)

#Converting w and b to tensors
w_t = tf.Variable(w, dtype = tf.float32, name = "w_t")
b_t = tf.Variable(b, dtype = tf.float32, name = "b_t")


def regress(X):
	# WRITEME: write your code here to complete the routine
        theta = tf.tuple([b_t,w_t])
	return tf.add(theta[0] , tf.multiply(X,theta[1]))

def gaussian_log_likelihood(mu, y):
	# WRITEME: write your code here to complete the sub-routine
        theta = tf.tuple([b_t,w_t])
	return (regress(mu) - y)
	
def computeCost(X, y,b_t,w_t): # loss is now Bernoulli cross-entropy/log likelihood
	#WRITEME: write your code here to complete the routine
    theta = tf.tuple([b_t,w_t])
    m = tf.cast(tf.shape(X), tf.float32)
    return (tf.divide(tf.reduce_sum(tf.square(gaussian_log_likelihood(X, y))), tf.multiply(2.0,m[0])))
	
def computeGrad(X, y,b_t,w_t):
    theta = tf.tuple([b_t,w_t])
    m = tf.cast(tf.shape(X), tf.float32)
    dL_db = tf.divide(tf.reduce_sum(gaussian_log_likelihood(X, y)), m[0]) # derivative w.r.t. model weights w
    dL_dw = tf.divide(tf.matmul(gaussian_log_likelihood(X, y), X, transpose_a = True), m[0]) # derivative w.r.t model bias b
    nabla = tf.tuple([dL_db, dL_dw]) # nabla represents the full gradient
    return nabla
    
    
computeCost = computeCost(X_p, y_p,b_t,w_t)
computeGrad = computeGrad(X_p , y_p,b_t,w_t)

cost = []

init = tf.global_variables_initializer()	
i = 0
with tf.Session() as sess:
    
    sess.run(init)
    #sess.graph.finalize() 
    
    while(i < n_epoch):
        L = sess.run([computeCost],feed_dict = {X_p:X , y_p:y })
        cost.append(L)
        print('{0} L = {1}'.format(i,L))
        i += 1
        grad_new1 = sess.run([computeGrad] , feed_dict = {X_p:X , y_p:y})
        print(((sess.run(w_t))))
        grad_new2 = [item for sublist in grad_new1 for item in sublist]
        print(((grad_new2[1])))
        b_t1 = tf.subtract(b_t ,tf.multiply(alpha,grad_new2[0]))
    	w_t1 = tf.subtract(w_t , tf.multiply(alpha,grad_new2[1]))
    	b_t2 = tf.convert_to_tensor(b_t1)
    	w_t2 = tf.convert_to_tensor(w_t1)
    	sess.run(tf.assign(b_t,b_t2))
    	sess.run(tf.assign(w_t,w_t2))

        
                
        
        
        
    print('W:', w_t)
    print('b:', b_t)
    theta = tf.tuple([b_t,w_t])
    kludge = 0.25 # helps with printing the plots (you can tweak this value if you like)
    # visualize the fit against the data
    X_test = np.linspace(data.X.min(), data.X.max(), 100)
    X_test = np.expand_dims(X_test, axis=1)
    X_test = np.float32(X_test)

    plt.figure(1)
    plt.plot(X_test, sess.run(regress(X_test)), label="Model")
    plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
    plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
    plt.legend(loc="best")
    plt.savefig(os.getcwd() + '/Pb_1_2_Test_Fig.jpg')
    
    
    # WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)


    # visualize the loss as a function of passes through the dataset
    # WRITEME: write your code here create and save a plot of loss versus epoch
    
    plt.figure(2)    
    plt.plot(cost)
    plt.title('Loss vs Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Cost')
    plt.savefig(os.getcwd() + '/Pb_1_2_Loss_Fig.jpg')
    
    
        
        
    plt.show() # convenience command to force plots to pop up on desktop


print("done")
    
       


