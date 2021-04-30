##################################################################
### This file is used to generate Figures & Tables in Chapter6 ###
##################################################################
#Note(Important): Since this file would require data from Data directory, 
#you will need to first run gendata.py before you run this file.

import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
from scipy.stats import beta
from tensorflow.random import set_seed
from scipy.stats import gaussian_kde as kde
import matplotlib.pyplot as plt

####### Plot Formatting ######
plt.rc('lines', linewidth = 4)
plt.rc('xtick', labelsize = 13)
plt.rc('ytick', labelsize = 13)
plt.rc('legend',fontsize=14)
plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['lines.markersize'] = 6
plt.rcParams['figure.figsize'] = (7.0, 5.0)

#### To make it cleaner, create Directory "images" to store all the figures ####
imagepath = os.path.join(os.getcwd(),"images")
os.makedirs(imagepath,exist_ok=True)

#######################################
#Define the activation function
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

##################################################
##### Feature Scaling and Data Preprocessing #####
##################################################

#### Before Scaling #####
a = 40
np.random.seed(12345)
listex = np.linspace(-a,a,100)
listey = listex**2 + 0.1*np.random.normal(0, 1, len(listex)) 

set_seed(12345)
model_rbf = tf.keras.Sequential()
model_rbf.add(tf.keras.layers.Dense(5,activation=rbf))
model_rbf.add(tf.keras.layers.Dense(1))
model_rbf.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model_rbf.fit(listex,listey, epochs=1000, verbose=0)

preds_rbf = []
size = 100
for j in np.linspace(-a,a,size):
    preds_rbf.append(model_rbf.predict([j]))
    
fig = plt.figure()
plt.xlabel('input')
plt.ylabel("output")
plt.scatter(listex, listey, marker='v',color='g', label="Training")     #size of point
plt.scatter(np.linspace(-a,a,size), np.array(preds_rbf), marker='p', color='r', label="Predict (RBF)")
plt.legend()
plt.show();
fig.savefig("images/NN_before_scale.png");

#### After Scaling #####
data = np.array((listex,listey)).T
scaler = MinMaxScaler((-1,1))
data_trans = scaler.fit_transform(data)
xmax, ymax = scaler.data_max_
xmin, ymin = scaler.data_min_

set_seed(12345)
model_rbf2 = tf.keras.Sequential()
model_rbf2.add(tf.keras.layers.Dense(5,activation=rbf))
model_rbf2.add(tf.keras.layers.Dense(1))
model_rbf2.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model_rbf2.fit(data_trans[:,0],data_trans[:,1],epochs=1000, verbose=0)

x_test = np.linspace(xmin,xmax,100).reshape(-1,1)
scaler = MinMaxScaler((-1,1))
x_test_trans = scaler.fit_transform(x_test) 
preds_rbf_trans = model_rbf2.predict(x_test_trans)
preds_rbf2 = (preds_rbf_trans+1)*(ymax-ymin)/2+ymin

fig = plt.figure()
plt.xlabel('input')
plt.ylabel("output")
plt.scatter(listex, listey, marker='v',color='g', label="Training Data")     #size of point
plt.scatter(x_test, preds_rbf2, marker='p', color='r', label="Predictions (RBF)")
plt.legend()
plt.show();
fig.savefig("images/NN_after_scale.png");

#############################################
##### Architecture of the hidden layers #####
#############################################
np.random.seed(12345)
x_act = np.pi*np.linspace(-1,1,200)
y_act = np.sin(x_act)+0.1*np.random.normal(0,1,len(x_act))

## RBF ##
set_seed(12345)
model_rbf_act = tf.keras.Sequential()
model_rbf_act.add(tf.keras.layers.Dense(5,activation=rbf))
model_rbf_act.add(tf.keras.layers.Dense(1))
model_rbf_act.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model_rbf_act.fit(x_act,y_act, epochs=1000, verbose=0)

## ReLU ##
set_seed(12345)
model_relu_act = tf.keras.Sequential()
model_relu_act.add(tf.keras.layers.Dense(5,activation='relu'))
model_relu_act.add(tf.keras.layers.Dense(1))
model_relu_act.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))   
model_relu_act.fit(x_act,y_act, epochs=1000, verbose=0)

## sigmoid ##
set_seed(12345)
model_sig_act = tf.keras.Sequential()
model_sig_act.add(tf.keras.layers.Dense(5,activation='sigmoid'))
model_sig_act.add(tf.keras.layers.Dense(1))
model_sig_act.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model_sig_act.fit(x_act,y_act, epochs=1000, verbose=0)

preds_rbf_act = []
preds_relu_act = []
preds_sig_act = []
size = 100
x_eval_act = np.pi*np.linspace(-1,1,size)
for j in x_eval_act:
    preds_rbf_act.append(model_rbf_act.predict([j]))
    preds_relu_act.append(model_relu_act.predict([j]))
    preds_sig_act.append(model_sig_act.predict([j]))

## RBF ##
p1 = tf.reshape(tf.constant(preds_rbf_act),len(preds_rbf_act))
error1 = tf.keras.losses.MSE(np.sin(x_eval_act),p1).numpy()
## ReLU ##
p2 = tf.reshape(tf.constant(preds_relu_act),len(preds_relu_act))
error2 = tf.keras.losses.MSE(np.sin(x_eval_act),p2).numpy()
## sigmoid ##
p3 = tf.reshape(tf.constant(preds_sig_act),len(preds_sig_act))
error3 = tf.keras.losses.MSE(np.sin(x_eval_act),p3).numpy()
print(error1, error2, error3)

#### Activated neurons when y=x^2 #####
########## Activation: ReLU  ##########
a = 40
np.random.seed(12345)
x_on1 = np.linspace(-1,1,100)
y_on1 = x_on1**2 + 0.1*np.random.normal(0, 1, len(x_on1)) 

num_neuron = 5
set_seed(12345)
model_relu_on = tf.keras.Sequential()
model_relu_on.add(tf.keras.layers.Dense(num_neuron,activation='relu'))
model_relu_on.add(tf.keras.layers.Dense(1))
model_relu_on.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
model_relu_on.fit(x_on1,y_on1, epochs=1500, verbose=0)

weight_relu_on = model_relu_on.get_weights()
bdry_relu_on = []
bdry1_relu_on = []
for i in range(weight_relu_on[0].shape[1]):
    val = -weight_relu_on[1][i]/weight_relu_on[0][0,i]
    if weight_relu_on[0][0,i] >0:
        bdry_relu_on.append((val,i))    
        bdry1_relu_on.append((val,i,'left'))     # left flat
    else:
        bdry_relu_on.append((val,i))   
        bdry1_relu_on.append((val,i,'right'))
        
bdry_arr_relu_on = np.array(sorted(bdry_relu_on))

def piece(weight,i,x):
    b1 = weight[1][i]
    w1 = weight[0][0,i]
    val = tf.keras.activations.relu(w1*x+b1).numpy()
    return (weight[2][i])*val

fig = plt.figure(figsize=(10,5))
plt.subplot(121)
x_rb = 1
x_lb = -1
for i in map(int,bdry_arr_relu_on[:,1]):
    if bdry1_relu_on[i][-1] == 'left' and bdry1_relu_on[i][0]<x_rb: 
        x = np.linspace(bdry1_relu_on[i][0],x_rb,20)
        plt.plot(x, [piece(weight_relu_on,i,val) for val in x], label='Neuron '+str(i), linewidth=4)
    elif bdry1_relu_on[i][-1] == 'right' and bdry1_relu_on[i][0]>x_lb:
        x = np.linspace(x_lb,bdry1_relu_on[i][0],20)
        plt.plot(x, [piece(weight_relu_on,i,val) for val in x], label='Neuron '+str(i), linewidth=4)
plt.xlim(-1,1)
plt.ylim(-0.5,1.5)
plt.title('After Scaling')
plt.legend()
plt.show();
        
plt.subplot(122)
preds_relu_on = []
size = 100
for j in np.linspace(-1,1,size):
    preds_relu_on.append(model_relu_on.predict([j]))
p = tf.reshape(tf.constant(preds_relu_on),len(preds_relu_on))
error = tf.keras.losses.MSE(x_on1**2,p).numpy()
plt.ylim(-0.5,1.5)
plt.title('MSE=%.8f'%(error))
plt.scatter(x_on1, y_on1, label="Training Data")     #size of point
plt.scatter(np.linspace(-1,1,size), np.array(preds_relu_on), s=10, label="Predictions (ReLU)")
plt.legend()
plt.show();
fig.savefig("images/NN_Neurons_on_relu.png")

#### Activated neurons when y=x^2 #####
########## Activation: RBF  ###########
num_neuron = 5

set_seed(12345)
model_rbf_on = tf.keras.Sequential()
model_rbf_on.add(tf.keras.layers.Dense(num_neuron,activation=rbf))
model_rbf_on.add(tf.keras.layers.Dense(1))
model_rbf_on.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
model_rbf_on.fit(x_on1,y_on1, epochs=1500, verbose=0)

weight_rbf_on = model_rbf_on.get_weights()
bdry_rbf_on = []

for i in range(weight_rbf_on[0].shape[1]):
    val1 = (-2-weight_rbf_on[1][i])/weight_rbf_on[0][0,i]
    val2 = (2-weight_rbf_on[1][i])/weight_rbf_on[0][0,i]
    bdry_rbf_on.append((min(val1,val2),max(val1,val2)))    

def overlap(interval1,interval2):
    if max(interval1[0],interval2[0]) < min(interval1[1],interval2[1]):
        return [max(interval1[0],interval2[0]), min(interval1[1],interval2[1])]
    else:
        return []
    
def piece_rbf(weight,i,x):
    b1 = weight[1][i]
    w1 = weight[0][0,i]
    val = rbf(w1*x+b1).numpy()
    return (weight[2][i])*val

fig = plt.figure(figsize=(10,5))
plt.subplot(121)
for i in range(num_neuron):
    interval = overlap([-1,1],bdry_rbf_on[i])
    if len(interval)!=0:
        x = np.linspace(interval[0],interval[1],20)
        plt.plot(x, [piece_rbf(weight_rbf_on,i,val) for val in x], label='Neuron '+str(i),linewidth=4)
plt.xlim(-1,1)
plt.ylim(-2,2)
plt.title('After Scaling')
plt.legend()
plt.show();

plt.subplot(122)
preds_rbf_on = []
size = 100
for j in np.linspace(-1,1,size):
    preds_rbf_on.append(model_rbf_on.predict([j]))
p = tf.reshape(tf.constant(preds_rbf_on),len(preds_rbf_on))
error_rbf_on = tf.keras.losses.MSE(x_on1**2,p).numpy()
plt.title('MSE=%.8f'%(error_rbf_on))
plt.scatter(x_on1, y_on1, label="Training Data")     #size of point
plt.scatter(np.linspace(-1,1,size), np.array(preds_rbf_on), s=10, label="Predictions (RBF)")
plt.legend()
plt.show();
fig.savefig("images/NN_Neurons_on_rbf.png")

#### Activated neurons when y=sin x #####
########### Activation: ReLU  ###########
np.random.seed(12345)
x_on2 = np.linspace(-1,1,100) 
y_on2 = np.sin(np.pi*x_on2) + 0.1*np.random.normal(0, 1, len(x_on2)) 

num_neuron = 5
set_seed(12345)
model_relu_sin = tf.keras.Sequential()
model_relu_sin.add(tf.keras.layers.Dense(num_neuron,activation='relu'))
model_relu_sin.add(tf.keras.layers.Dense(1))
model_relu_sin.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
model_relu_sin.fit(x_on2,y_on2, epochs=1500, verbose=0)

weight_sin = model_relu_sin.get_weights()
bdry_sin = []
bdry1_sin = []
for i in range(weight_sin[0].shape[1]):
    val = -weight_sin[1][i]/weight_sin[0][0,i]
    if weight_sin[0][0,i] >0:
        bdry_sin.append((val,i))    
        bdry1_sin.append((val,i,'left'))     
    else:
        bdry_sin.append((val,i))   
        bdry1_sin.append((val,i,'right'))
        
bdry_arr_sin = np.array(sorted(bdry_sin))

def piece(weight,i,x):
    b1 = weight[1][i]
    w1 = weight[0][0,i]
    val = tf.keras.activations.relu(w1*x+b1).numpy()
    return (weight[2][i])*val

fig = plt.figure(figsize=(10,5))
plt.subplot(121)
x_rb = 1
x_lb = -1
for i in map(int,bdry_arr_sin[:,1]):
    if bdry1_sin[i][-1] == 'left' and bdry1_sin[i][0]<x_rb: 
        x = np.linspace(bdry1_sin[i][0],x_rb,20)
        plt.plot(x, [piece(weight_sin,i,val) for val in x], label='Neuron '+str(i), linewidth=4)
    elif bdry1_sin[i][-1] == 'right' and bdry1_sin[i][0]>x_lb:
        x = np.linspace(x_lb,bdry1_sin[i][0],20)
        plt.plot(x, [piece(weight_sin,i,val) for val in x], label='Neuron '+str(i), linewidth=4)
plt.xlim(-1,1)
plt.ylim(-1.5,1.5)
plt.title('After Scaling')
plt.legend()
plt.show();

plt.subplot(122)
preds_relu_sin = []
size = 100
for j in np.linspace(-1,1,size):
    preds_relu_sin.append(model_relu_sin.predict([j]))
p = tf.reshape(tf.constant(preds_relu_sin),len(preds_relu_sin))
error = tf.keras.losses.MSE(np.sin(np.pi*x_on2),p).numpy()
plt.title('MSE=%.8f'%(error))
plt.axhline(0,c='b',linestyle=':')
plt.ylim(-1.5,1.5)
plt.scatter(x_on2, y_on2, label="Trainings Data")     #size of point
plt.scatter(np.linspace(-1,1,size), np.array(preds_relu_sin), s=10, label="Predictions (ReLU)")
plt.legend()
plt.show();
fig.savefig("images/NN_Neurons_on_sin_relu.png");

#### Activated neurons when y=sin x #####
########### Activation: RBF  ############
set_seed(12345)
model_rbf_sin = tf.keras.Sequential()
model_rbf_sin.add(tf.keras.layers.Dense(2,activation=rbf))
model_rbf_sin.add(tf.keras.layers.Dense(1))
model_rbf_sin.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
model_rbf_sin.fit(x_on2,y_on2, epochs=1500, verbose=0)

weight_rbf_sin = model_rbf_sin.get_weights()
bdry_sin = []

for i in range(weight_rbf_sin[0].shape[1]):
    val1 = (-2-weight_rbf_sin[1][i])/weight_rbf_sin[0][0,i]
    val2 = (2-weight_rbf_sin[1][i])/weight_rbf_sin[0][0,i]
    bdry_sin.append((min(val1,val2),max(val1,val2)))    

def overlap(interval1,interval2):
    if max(interval1[0],interval2[0]) < min(interval1[1],interval2[1]):
        return [max(interval1[0],interval2[0]), min(interval1[1],interval2[1])]
    else:
        return []
    
def piece_rbf(weight,i,x):
    b1 = weight[1][i]
    w1 = weight[0][0,i]
    val = rbf(w1*x+b1).numpy()
    return (weight[2][i])*val

fig = plt.figure(figsize=(10,5))
plt.subplot(121)
for i in range(2):
    interval = overlap([-1,1],bdry_sin[i])
    if len(interval)!=0:
        x = np.linspace(interval[0],interval[1],20)
        plt.plot(x, [piece_rbf(weight_rbf_sin,i,val) for val in x], label='Neuron '+str(i),linewidth=4)
plt.xlim(-1,1)
plt.ylim(-2,2)
plt.axhline(0,c='b',linestyle=':')
plt.title('After Scaling')
plt.legend()
plt.show();

plt.subplot(122)
preds_rbf_sin = []
size = 100
for j in np.linspace(-1,1,size):
    preds_rbf_sin.append(model_rbf_sin.predict([j]))
p = tf.reshape(tf.constant(preds_rbf_sin),len(preds_rbf_sin))
error = tf.keras.losses.MSE(np.sin(np.pi*x_on2),p).numpy()
plt.title('MSE=%.8f'%(error))
plt.axhline(0,c='b',linestyle=':')
plt.scatter(x_on2, y_on2, label="Trainings Data")     
plt.scatter(np.linspace(-1,1,size), np.array(preds_rbf_sin), s=10, label="predictions (RBF)")
plt.ylim(-2,2)
plt.legend()
plt.show();
fig.savefig("images/NN_Neurons_on_sin_rbf.png");

#############################################
############## Cross-Validation #############
##### y=sin x; Training size = 1000 #####
data_dir = "Data_Chp6"
epo_ls = 10*np.arange(1,11)
hid_ls = np.arange(2,11) #1 is not good
score_all = np.zeros((len(hid_ls),len(epo_ls)))
ratio = [0.0, 1.0]
for i in range(len(hid_ls)):
    filename = '%s/sin_1%s1_split_MSE.mat'%(data_dir,hid_ls[i])
    partial_data = sio.loadmat(filename)
    score_all[i] = ratio[0]*partial_data['train'] + ratio[1]*partial_data['test']    

fig = plt.figure(figsize=(10,5))
for i in range(len(hid_ls)):
    plt.semilogy(epo_ls, score_all[i,:], label='%d'%(hid_ls[i]))
plt.xticks(epo_ls)
plt.xlabel('Number of epochs')
plt.ylabel('Error ($y=\sin x$)')
plt.title('Error = %.1f*MSE_train+%.1f*MSE_test'%(ratio[0],ratio[1]))
plt.legend()
plt.show();
fig.savefig("images/NN_sin0.png");

##### y=sin x; Training size = 5000 #####
## Training size = 1000, ratio = (0.5,0.5)
epo_ls = 10*np.arange(1,11)
hid_ls = np.arange(2,11) #1 is not good
score_all = np.zeros((len(hid_ls),len(epo_ls)))
ratio = [0.5, 0.5]
for i in range(len(hid_ls)):
    filename = '%s/sin5_1%s1_split_MSE.mat'%(data_dir,hid_ls[i])
    partial_data = sio.loadmat(filename)
    score_all[i] = ratio[0]*partial_data['train'] + ratio[1]*partial_data['test'] 

fig = plt.figure(figsize=(10,5))
for i in range(len(hid_ls)):
    plt.semilogy(epo_ls, score_all[i,:], label='%d'%(hid_ls[i]))
plt.xticks(epo_ls)
plt.xlabel('Number of epochs')
plt.ylabel('Error ($y=\sin x$)')
plt.title('Error = %.1f*MSE_train+%.1f*MSE_test'%(ratio[0],ratio[1]))
plt.legend()
plt.show();
fig.savefig("images/NN_sin1.png");

## Training size = 1000, ratio = (0.0,1.0)
epo_ls = 10*np.arange(1,11)
hid_ls = np.arange(2,11) #1 is not good
score_all = np.zeros((len(hid_ls),len(epo_ls)))
ratio = [0.0, 1.0]
for i in range(len(hid_ls)):
    filename = '%s/sin5_1%s1_split_MSE.mat'%(data_dir,hid_ls[i])
    partial_data = sio.loadmat(filename)
    score_all[i] = ratio[0]*partial_data['train'] + ratio[1]*partial_data['test']

fig = plt.figure(figsize=(10,5))
for i in range(len(hid_ls)):
    plt.semilogy(epo_ls, score_all[i,:], label='%d'%(hid_ls[i]))
plt.xticks(epo_ls)
plt.xlabel('Number of epochs')
plt.ylabel('Error ($y=\sin x$)')
plt.title('Error = %.1f*MSE_train+%.1f*MSE_test'%(ratio[0],ratio[1]))
plt.legend()
plt.show();
fig.savefig("images/NN_sin2.png");

##### y=(x+3)(x-1)^2; Training size = 1000 #####
epo_ls = 100*np.arange(1,11)
hid_ls = np.arange(2,11) #1 is not good
score_all = np.zeros((len(hid_ls),len(epo_ls)))
ratio = [0.5, 0.5]

for i in range(len(hid_ls)):
    filename = '%s/poly_1%s1_split_MSE.mat'%(data_dir,hid_ls[i])
    partial_data = sio.loadmat(filename)
    score_all[i] = ratio[0]*partial_data['train'] + ratio[1]*partial_data['test'] 

fig = plt.figure(figsize=(10,5))
for i in range(len(hid_ls)):
    plt.semilogy(epo_ls, score_all[i,:], label='%d'%(hid_ls[i]))
plt.xticks(epo_ls)
plt.xlabel('Number of epochs')
plt.ylabel('Error (Polynomial)')
plt.title('Error = %.1f*MSE_train+%.1f*MSE_test'%(ratio[0],ratio[1]))
plt.legend()
plt.show();
fig.savefig("images/NN_poly1.png");

##########################################################
##### Neural networks in a data-consistent framework #####
##########################################################
def Qexact(l):
    return np.exp(-l*0.5)

np.random.seed(12345)
size = 100
delta_t = 0.01
n = int(0.5/delta_t)
lam_in = np.random.normal(0, 1, size)
y_out = np.array([(1-i*delta_t)**n for i in lam_in])

num_neuron = 5
tf.random.set_seed(12345)
model_ode = tf.keras.Sequential()
model_ode.add(tf.keras.layers.Dense(num_neuron,activation=rbf))
model_ode.add(tf.keras.layers.Dense(1))

model_ode.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
model_ode.fit(lam_in,y_out, epochs=1500, verbose=0)

###########################
##### Forward Problem #####
#Build $\pi_D^Q$ and $\pi_D^{Qhat}$, use 1000 samples
N_kde = int(1E3)
np.random.seed(12345)
init_sample = np.random.normal(size = N_kde)
pfinit_sample = Qexact(init_sample)
pfinit_dens = kde(pfinit_sample)  #pf_exact
NN_sample = []
for j in init_sample:
    NN_sample.append(model_ode.predict([j])[0,0])
NN_dens = kde(NN_sample)    #pf_approx

fig = plt.figure(figsize=(8,6))
x = np.linspace(-3,3,100)
plt.plot(x,pfinit_dens(x),'--',linewidth=6,color='r',label='Exact Push-forward')
plt.plot(x,NN_dens(x),color='b',label='Approximate Push-forward')   
plt.xlabel('$\mathcal{D}$')
plt.legend()
plt.show();
fig.savefig('images/NN_forward.png');

###########################
##### Inverse Problem #####
N_inv = int(1E4)

## Use true samples to get obs dens
exact_sample = beta.rvs(2, 5, loc=0, scale=1, size=N_inv, random_state=12345)
pfexact_sample = Qexact(exact_sample)
obs_dens = kde(pfexact_sample)

## exact and approx pf of initial dens
np.random.seed(12345)
init_sample_inv = np.random.normal(size=N_inv)
NN_sample = []
for j in init_sample_inv:
    NN_sample.append(model_ode.predict([j])[0,0])
NN_dens = kde(NN_sample)   ## approx pf

## updated sample using rejection sampling
r = np.array([obs_dens(i)/NN_dens(i) for i in NN_sample]).flatten()
M = max(r)
np.random.seed(12345)
unif = np.random.uniform(size=len(r))
idx = list(M*unif < r)
up_sample = init_sample_inv[idx]

up_dens = kde(up_sample)

fig = plt.figure(figsize=(8,6))
size = 100
x = np.linspace(-1,1,size)
plt.plot(x,beta.pdf(x, 2, 5, loc=0, scale=1),'--',linewidth=6,color='r',label='Exact')
plt.plot(x,up_dens(x),color='b',label='Updated')   
plt.xlabel('$\Lambda$')
plt.legend()
plt.show();
fig.savefig('images/NN_inverse.png');