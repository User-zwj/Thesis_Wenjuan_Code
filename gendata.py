######################################################################
### This file is used to generate Data needed for the ODE example  ###
######################################################################

import os 
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.random import set_seed
from sklearn.model_selection import KFold


#######################################
#define the activation function
def rbf(x):
    return tf.math.exp(-x**2)
def d_rbf(x):
    return tf.gradients(rbf,x)
def rbf_grad(op, grad):
    x = op.inputs[0]
    n_gr = d_rbf(x)    #defining the gradient.
    return grad * n_gr
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+2))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)   
def tf_rbf(x,name=None):
    with tf.name_scope(name, "rbf", [x]) as name:
        y = py_func(rbf,   #forward pass function
                    [x],
                    [tf.float32],
                    name=name,
                    grad= rbf_grad) #the function that overrides gradient
        y[0].set_shape(x.get_shape())     #when using with the code, it is used to specify the rank of the input.
    return y[0]


def func(X,y,n_fold,n_hid,n_epo,seed=12345):
    '''
    return the average of weighted error for `n_fold` splits
    '''
    train_score = []
    test_score = []
    set_seed(seed)
    kf = KFold(n_splits=n_fold)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(n_hid,activation=rbf,input_dim=1))  #hid2
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
        model.fit(X_train,y_train, epochs=n_epo, verbose=0)

        error_train = tf.keras.losses.MSE(model.predict(X_train).flatten(),y_train).numpy()
        error_test = tf.keras.losses.MSE(model.predict(X_test).flatten(),y_test).numpy()
        train_score.append(error_train)
        test_score.append(error_test)
    return np.mean(train_score), np.mean(test_score)


#### To make it cleaner, create Directory "Data_Chp6" to store all the data ####
data_dir = "Data_Chp6"
datapath = os.path.join(os.getcwd(),data_dir)
os.makedirs(datapath,exist_ok=True)

hid_ls = np.arange(1,11)
n_fold = 5

##############################################################
##### Generate datafiles Data/sin_1[1-10]1_split_MSE.mat #####
##############################################################
epo_ls = 10*np.arange(1,11)
np.random.seed(12345)
X = np.pi*np.random.uniform(-1,1,size=1000)
y = np.sin(X)
st_name = "sin"
for i in range(len(hid_ls)):
    score = [func(X,y,n_fold,hid_ls[i],j) for j in epo_ls]  
    train = [score[i][0] for i in range(len(score))]
    test = [score[i][1] for i in range(len(score))]
    filename = '%s/%s_1%s1_split_MSE.mat'%(data_dir,st_name,hid_ls[i]) 
    data_dict = {'train':train, 'test':test}
    sio.savemat(filename, data_dict)

###############################################################
##### Generate datafiles Data/sin5_1[1-10]1_split_MSE.mat #####
###############################################################
epo_ls = 10*np.arange(1,11)
np.random.seed(12345)
X = np.pi*np.random.uniform(-1,1,size=5000)
y = np.sin(X)
st_name = "sin5"
for i in range(len(hid_ls)):
    score = [func(X,y,n_fold,hid_ls[i],j) for j in epo_ls]  
    train = [score[i][0] for i in range(len(score))]
    test = [score[i][1] for i in range(len(score))]
    filename = '%s/%s_1%s1_split_MSE.mat'%(data_dir,st_name,hid_ls[i]) 
    data_dict = {'train':train, 'test':test}
    sio.savemat(filename, data_dict)

###############################################################
##### Generate datafiles Data/poly_1[1-10]1_split_MSE.mat #####
###############################################################
epo_ls = 100*np.arange(1,11)
np.random.seed(12345)
X = np.random.uniform(-3,3,size=1000)
y = (X+3)*(X-1)**2
st_name = "poly"
for i in range(len(hid_ls)):
    score = [func(X,y,n_fold,hid_ls[i],j) for j in epo_ls]  
    train = [score[i][0] for i in range(len(score))]
    test = [score[i][1] for i in range(len(score))]
    filename = '%s/%s_1%s1_split_MSE.mat'%(data_dir,st_name,hid_ls[i]) 
    data_dict = {'train':train, 'test':test}
    sio.savemat(filename, data_dict)
    
###############################################################
##### Generate datafiles Data/poly5_1[1-10]1_split_MSE.mat #####
###############################################################
epo_ls = 100*np.arange(1,11)
np.random.seed(12345)
X = np.random.uniform(-3,3,size=5000)
y = (X+3)*(X-1)**2
st_name = "poly5"
for i in range(len(hid_ls)):
    score = [func(X,y,n_fold,hid_ls[i],j) for j in epo_ls]  
    train = [score[i][0] for i in range(len(score))]
    test = [score[i][1] for i in range(len(score))]
    filename = '%s/%s_1%s1_split_MSE.mat'%(data_dir,st_name,hid_ls[i]) 
    data_dict = {'train':train, 'test':test}
    sio.savemat(filename, data_dict)