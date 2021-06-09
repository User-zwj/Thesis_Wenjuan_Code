##################################################################
### This file is used to generate Figures & Tables in Chapter7 ###
##################################################################

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.random import set_seed
from math import factorial
from scipy.stats import norm
from scipy.integrate import odeint
import numpy.polynomial.hermite_e as H 
from sklearn.preprocessing import StandardScaler
import dolfin as fn
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt

####### Plot Formatting ######
plt.rc('lines', linewidth = 4)
plt.rc('xtick', labelsize = 13)
plt.rc('ytick', labelsize = 13)
plt.rc('legend',fontsize=14)
plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['lines.markersize'] = 8
plt.rcParams['figure.figsize'] = (7.0, 5.0)

#### To make it cleaner, create Directory "images" to store all the figures ####
imagepath = os.path.join(os.getcwd(),"images")
os.makedirs(imagepath,exist_ok=True)

##################################################
##############        PCE vs MC     ##############
##################################################

############## PCE ###############
start_def = time.time()
def Phi(n):
    #define H_n
    coeffs = [0]*(n+1)
    coeffs[n] = 1
    return coeffs

def inner2_herm(n):       ###return the denominator when computing $k_i$
    return factorial(n)

def product3_herm(i,j,l):
    #compute \Phi_i*\Phi_j*\Phi_l
    return lambda x: H.hermeval(x, H.hermemul(H.hermemul(Phi(i),Phi(j)),Phi(l))) 

def inner3_herm(P,i,j,l):
    #compute <\Phi_i\Phi_j\Phi_l>
    #Set up Gauss-Hermite quadrature, weighting function is exp^{-x^2}
    m=(P+1)**2
    x, w=H.hermegauss(m)        
    inner=sum([product3_herm(i,j,l)(x[idx]) * w[idx] for idx in range(m)])               return inner/np.sqrt(2*np.pi)   #because of the weight
time_def = time.time() - start_def

start_prep = time.time()
P=4
ki_herm = [0,1]+[0]*(P-1)
Inner3_herm = np.zeros((P+1,P+1,P+1)) #store all inner3_herm values
Inner2_herm = np.zeros(P+1)
for i in range(P+1):
    for j in range(P+1):
        for l in range(P+1):
            Inner3_herm[i,j,l] = inner3_herm(P,i,j,l)
for i in range(P+1):
    Inner2_herm[i] = inner2_herm(i)
time_prep = time.time() - start_prep

start_ode = time.time()
def ode_system_herm(y, t, P):   
    #P indicates the highest degree
    dydt = np.zeros(P+1) 
    for l in range(len(dydt)):
        dydt[l] = -(sum(sum(Inner3_herm[i,j,l]*ki_herm[i]*y[j] for j in range(P+1)) for i in range(P+1)))/Inner2_herm[l]
    return dydt
time_ode = time.time() - start_ode

start_solveode = time.time()
sol_herm = odeint(ode_system_herm, [1.0]+[0.0]*P, np.linspace(0,1,101), args=(P, )) 
time_solveode = time.time() - start_solveode

time_all = time_def + time_prep + time_ode + time_solveode

############## MC ###############
start_ode_mc = time.time()
def ode(y,t,nsample,k):
    '''
    Build the ode system
    '''
    dydt = np.zeros(nsample)
    for i in range(nsample):
        dydt[i] = -k[i]*y[i]
    return dydt
time_def_mc = time.time() - start_ode_mc

nsample = np.array([10, 100, 1000, 10000, 100000])
time_solveode_mc = np.zeros(len(nsample))
start_solveode_mc = np.zeros(len(nsample))
mean_mc_1 = np.zeros(len(nsample))
mean_mc_05 = np.zeros(len(nsample))
for i in range(len(nsample)):
    k = norm.rvs(loc=0, scale=1, size=nsample[i], random_state=12345)
    start_solveode_mc[i] = time.time()
    sol_mc = odeint(ode, [1.0]*nsample[i], np.linspace(0,1,101),args=(nsample[i],k))  #t:np.linspace(0,1,101)
    mean_mc_1[i] = np.mean(sol_mc[100,:])
    mean_mc_05[i] = np.mean(sol_mc[50,:])
    time_solveode_mc[i] = time.time() - start_solveode_mc[i]

time_all_mc = time_def_mc + time_solveode_mc

############## Comparison ###############
#### Table 7.2, row 1 ####
### PCE
print(time_solveode)     
### MC
print(time_solveode_mc)

## t = 0.5
mean_pc_05 = sol_herm[:,0][50]   #mean value using pc at t=0.5
mean_exact_05 = np.e**(1/8)
## t = 1
mean_pc_1 = sol_herm[:,0][100]   #mean value using pc at t=1
mean_exact_1 = np.e**(1/2)

#### Table 7.2, row 2 ####
print(mean_pc_05)
print(mean_mc_05)
print(mean_exact_05)

print()

#### Table 7.2, row 3 ####
print(mean_pc_1)
print(mean_mc_1)
print(mean_exact_1)


##################################################
##############       NN vs Poly     ##############
##################################################

############## NN ###############
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

np.random.seed(12345)
size = 100
delta_t = 0.01
n = int(0.5/delta_t)
### Original data ###
lam_in = np.random.normal(0, 1, size)
y_exact = np.array([np.exp(-i*0.5) for i in lam_in])
y_out = np.array([(1-i*delta_t)**n for i in lam_in])
### After feature scaling ###
scaler = StandardScaler()
data_trans = scaler.fit_transform(lam_in.reshape(-1,1))

num_neuron = 5
tf.random.set_seed(12345)
model_ode = tf.keras.Sequential()
model_ode.add(tf.keras.layers.Dense(num_neuron,activation=rbf))
model_ode.add(tf.keras.layers.Dense(1))
model_ode.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
model_ode.fit(data_trans[:,0],y_out, epochs=1500, verbose=0)

preds_ode = []
for j in data_trans[:,0]:
    preds_ode.append(model_ode.predict([j]))
preds_ode_shaped = tf.reshape(tf.constant(np.array(preds_ode)),len(preds_ode))
mse_fd_nn = tf.keras.losses.MSE(y_exact,preds_ode_shaped).numpy()

fig = plt.figure()
plt.xlabel("$\lambda$")
plt.ylabel("$q$")
plt.title("MSE=%.5f"%(mse_fd_nn))
plt.scatter(lam_in, y_out, label='Obs')
plt.scatter(lam_in, preds_ode_shaped, label='NN')
plt.legend()
plt.show();
# fig.savefig("images/comp_fd_nn.png")

############## Polynomial Regression ###############
mymodel3 = np.poly1d(np.polyfit(data_trans[:,0], y_out, 3))
preds_fd_pr3 = mymodel3(data_trans[:,0])
mse_fd_pr3 = tf.keras.losses.MSE(y_exact,preds_fd_pr3).numpy()

fig = plt.figure()
plt.xlabel("$\lambda$")
plt.ylabel("$q$")
plt.title("MSE=%.5f"%(mse_fd_pr3))
plt.scatter(lam_in, y_out, label='Obs')
plt.scatter(lam_in, preds_fd_pr3, label='PR (deg=3)')
plt.legend()
plt.show();
# fig.savefig("images/comp_fd_pr3.png")


####################################################
########  Stochastic collocation method   ##########
####################################################
def QoI_FEM(x0,y0,lam1,lam2,gridx,gridy,p):
    mesh = fn.UnitSquareMesh(gridx, gridy)
    V = fn.FunctionSpace(mesh, "Lagrange", p)

    # Define diffusion tensor (here, just a scalar function) and parameters
    A = fn.Expression((('exp(lam1)','a'),
                ('a','exp(lam2)')), a = fn.Constant(0.0), lam1 = lam1, lam2 = lam2, degree=3) 

    u_exact = fn.Expression("sin(lam1*pi*x[0])*cos(lam2*pi*x[1])", lam1 = lam1, lam2 = lam2, degree=2+p)

    # Define the mix of Neumann and Dirichlet BCs
    class LeftBoundary(fn.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] < fn.DOLFIN_EPS)
    class RightBoundary(fn.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] > 1.0 - fn.DOLFIN_EPS)
    class TopBoundary(fn.SubDomain):
        def inside(self, x, on_boundary):
            return (x[1] > 1.0 - fn.DOLFIN_EPS)
    class BottomBoundary(fn.SubDomain):
        def inside(self, x, on_boundary):
            return (x[1] < fn.DOLFIN_EPS)

    # Create a mesh function (mf) assigning an unsigned integer ('uint')
    # to each edge (which is a "Facet" in 2D)
    mf = fn.MeshFunction('size_t', mesh, 1)
    mf.set_all(0) # initialize the function to be zero
    # Setup the boundary classes that use Neumann boundary conditions
    NTB = TopBoundary() # instatiate
    NTB.mark(mf, 1) # set all values of the mf to be 1 on this boundary
    NBB = BottomBoundary()
    NBB.mark(mf, 2) # set all values of the mf to be 2 on this boundary
    NRB = RightBoundary()
    NRB.mark(mf, 3)

    # Define Dirichlet boundary conditions
    Gamma_0 = fn.DirichletBC(V, u_exact, LeftBoundary())
    bcs = [Gamma_0]

    # Define data necessary to approximate exact solution
    f = ( fn.exp(lam1)*(lam1*fn.pi)**2 + fn.exp(lam2)*(lam2*fn.pi)**2 ) * u_exact
    g1 = fn.Expression("-exp(lam2)*lam2*pi*sin(lam1*pi*x[0])*sin(lam2*pi*x[1])", lam1=lam1, lam2=lam2, degree=2+p)    #pointing outward unit normal vector, pointing upaward (0,1)
    g2 = fn.Expression("exp(lam2)*lam2*pi*sin(lam1*pi*x[0])*sin(lam2*pi*x[1])", lam1=lam1, lam2=lam2, degree=2+p)     #pointing downward (0,1)
    g3 = fn.Expression("exp(lam1)*lam1*pi*cos(lam1*pi*x[0])*cos(lam2*pi*x[1])", lam1=lam1, lam2=lam2, degree=2+p)

    fn.ds = fn.ds(subdomain_data=mf)
    # Define variational problem
    u = fn.TrialFunction(V)
    v = fn.TestFunction(V)
    a = fn.inner(A*fn.grad(u), fn.grad(v))*fn.dx
    L = f*v*fn.dx + g1*v*fn.ds(1) + g2*v*fn.ds(2) + g3*v*fn.ds(3)  #note the 1, 2 and 3 correspond to the mf

    # Compute solution
    u = fn.Function(V)
    fn.solve(a == L, u, bcs)

    return u(x0,y0)

def exactQ(x,y):
    return (1-np.cos(np.pi*x))*np.sin(np.pi*y)/(np.pi**2*x*y)

x0 = [0.2, 0.2, 0.2, 0.3, 0.5, 0.5]
y0 = [0.3, 0.5, 0.8, 0.2, 0.2, 0.5]
M, N = 5, 5
x1,w1 = leggauss(M)
x2,w2 = leggauss(N)
tab = np.zeros((len(x0),2))
for k in range(len(x0)):
    #### Stochastic Collocation Mean at x0, y0 ####
    uij = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            uij[i,j] = QoI_FEM(x0[k],y0[k],(1+x1[i])/2,(1+x2[j])/2,10,10,2)
            
    sol = 0
    for i in range(M):
        for j in range(N):
            sol += w1[i]*w2[j]*uij[i,j]
    sol /= 4
    tab[k,0] = sol
    #### Exact Mean at x0, y0 ####
    tab[k,1] = exactQ(x0[k],y0[k])

print(tab)

##################################################
##############       Discussion     ##############
##################################################
def model(x):
    if x<=1:
        return 15*x+10
    elif x<=7:
        return x**3-12*x**2+36*x
    elif x<=10:
        return 15/np.pi*np.sin(np.pi*(x-7))+7
    else:
        return -30*np.sqrt(x-9)+37

np.random.seed(12345)
x_syn = np.random.uniform(0,15,100)
y_exact = np.array([model(i) for i in x_syn])
y_syn = y_exact+np.random.normal(0,1,len(x_syn))

scaler_syn = StandardScaler()
syndata_trans = scaler_syn.fit_transform(x_syn.reshape(-1,1))

num_neuron = 5
tf.random.set_seed(12345)
model_syn_all = tf.keras.Sequential()
model_syn_all.add(tf.keras.layers.Dense(num_neuron,activation=rbf))
model_syn_all.add(tf.keras.layers.Dense(num_neuron,activation=rbf))
# model_syn_all.add(tf.keras.layers.Dense(num_neuron,activation=rbf))
# model_syn_all.add(tf.keras.layers.Dense(num_neuron,activation=rbf))
model_syn_all.add(tf.keras.layers.Dense(1))
model_syn_all.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
model_syn_all.fit(syndata_trans[:,0],y_syn, epochs=1000, verbose=0)

fig = plt.figure()
plt.xlabel("$x$")
plt.ylabel("$y$")
y_pred = []
for j in syndata_trans[:,0]:
    y_pred.append(model_syn_all.predict([j]))
y_pred_shaped = tf.reshape(tf.constant(np.array(y_pred)),len(y_pred))
error0 = tf.keras.losses.MSE(y_syn,y_pred_shaped).numpy()
plt.title("MSE=%.5f"%(error0))
plt.scatter(x_syn, y_syn, label='Obs')
plt.scatter(x_syn, y_pred_shaped, color='red',label="NN")
plt.legend()
plt.show();
# fig.savefig("images/comp_nn_1step.png")


## Step 1
num_neuron = 5
tf.random.set_seed(12345)
model_nn1 = tf.keras.Sequential()
model_nn1.add(tf.keras.layers.Dense(num_neuron,activation=rbf))
model_nn1.add(tf.keras.layers.Dense(1))
model_nn1.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
model_nn1.fit(syndata_trans[:,0],y_syn, epochs=1000, verbose=0)

fig = plt.figure(figsize=(13,4))
plt.subplot(121)
plt.xlabel("$x$")
plt.ylabel("$y$")
y_pred1 = []
for j in syndata_trans[:,0]:
    y_pred1.append(model_nn1.predict([j]))
y_pred1_shaped = tf.reshape(tf.constant(np.array(y_pred1)),len(y_pred1))
plt.title("Overall Fit")
plt.scatter(x_syn, y_syn, label='Obs')
plt.scatter(x_syn, y_pred1_shaped, color='red',label="NN")
plt.legend();
plt.subplot(122)
plt.xlabel("$x$")
plt.ylabel("Residual")
mse1 = tf.keras.losses.MSE(y_syn,y_pred1_shaped).numpy()
plt.title("MSE=%.5f"%(mse1))
plt.scatter(x_syn, y_syn-y_pred1_shaped)
plt.show();
# fig.savefig("images/comp_nn_step1.png");

## Step 2
tf.random.set_seed(12345)
model_nn2 = tf.keras.Sequential()
model_nn2.add(tf.keras.layers.Dense(5,activation=rbf))
model_nn2.add(tf.keras.layers.Dense(1))
error1 = y_syn - y_pred1_shaped
scaler_syn2 = StandardScaler()
model_nn2.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
model_nn2.fit(syndata_trans[:,0],error1, epochs=1000, verbose=0)

fig = plt.figure(figsize=(15,4))
plt.subplot(121)
plt.xlabel("$x$")
plt.ylabel("$y$")
y_pred2 = []
for j in syndata_trans[:,0]:
    y_pred2.append(model_nn2.predict([j]))
y_pred2_shaped = tf.reshape(tf.constant(np.array(y_pred2)),len(y_pred2))
plt.title("Overall Fit")
plt.scatter(x_syn, y_syn, label='Obs')
plt.scatter(x_syn, y_pred1_shaped+y_pred2_shaped, color='red',label="NN")
plt.legend();
plt.subplot(122)
plt.xlabel("$x$")
plt.ylabel("Residual")
mse2 = tf.keras.losses.MSE(y_syn,y_pred1_shaped+y_pred2_shaped).numpy()
plt.title("MSE=%.5f"%(mse2))
plt.scatter(x_syn, y_syn-y_pred1_shaped-y_pred2_shaped)
plt.show();
# fig.savefig("images/comp_nn_step2.png");

## Step 3
tf.random.set_seed(12345)
model_nn3 = tf.keras.Sequential()
model_nn3.add(tf.keras.layers.Dense(5,activation=rbf))
model_nn3.add(tf.keras.layers.Dense(1))
error2 = y_syn - y_pred1_shaped - y_pred2_shaped
scaler_syn3 = StandardScaler()
model_nn3.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
model_nn3.fit(syndata_trans[:,0],error2, epochs=1000, verbose=0)

fig = plt.figure(figsize=(13,4))
plt.subplot(121)
plt.xlabel("$x$")
plt.ylabel("$y$")
y_pred3 = []
for j in syndata_trans[:,0]:
    y_pred3.append(model_nn3.predict([j]))
y_pred3_shaped = tf.reshape(tf.constant(np.array(y_pred3)),len(y_pred3))
plt.title("Overall Fit")
plt.scatter(x_syn, y_syn, label='Obs')
plt.scatter(x_syn, y_pred1_shaped+y_pred2_shaped+y_pred3_shaped, color='red',label="NN")
plt.legend();
plt.subplot(122)
plt.xlabel("$x$")
plt.ylabel("Residual")
mse3 = tf.keras.losses.MSE(y_syn,y_pred1_shaped+y_pred2_shaped+y_pred3_shaped).numpy()
plt.title("MSE=%.5f"%(mse3))
plt.scatter(x_syn, y_syn-y_pred1_shaped-y_pred2_shaped-y_pred3_shaped);
plt.show();
# fig.savefig("images/comp_nn_step3.png");

## all in one
num_neuron = 15
tf.random.set_seed(12345)
model_nn11 = tf.keras.Sequential()
model_nn11.add(tf.keras.layers.Dense(num_neuron,activation=rbf))
model_nn11.add(tf.keras.layers.Dense(1))
model_nn11.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
model_nn11.fit(syndata_trans[:,0],y_syn, epochs=5000, verbose=0)

fig = plt.figure(figsize=(13,4))
plt.subplot(121)
plt.xlabel("$x$")
plt.ylabel("$y$")
y_pred11 = []
for j in syndata_trans[:,0]:
    y_pred11.append(model_nn11.predict([j]))
y_pred11_shaped = tf.reshape(tf.constant(np.array(y_pred11)),len(y_pred11))
plt.title("Overall Fit")
plt.scatter(x_syn, y_syn, label='Obs')
plt.scatter(x_syn, y_pred11_shaped, color='red', label="NN")
plt.legend();
plt.subplot(122)
plt.xlabel("$x$")
plt.ylabel("Residual")
mse11 = tf.keras.losses.MSE(y_syn,y_pred11_shaped).numpy()
plt.title("MSE=%.5f"%(mse11))
plt.scatter(x_syn, y_syn-y_pred11_shaped)
plt.show();
# fig.savefig("images/comp_nn_step11.png");