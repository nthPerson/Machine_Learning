# Utility functions used for HA4

import numpy as np
import os
import sys

def load_data():
    data_files = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']
    for df in data_files:
        if not os.path.exists(df):
            sys.stderr.write('Make sure that {} is in the current directory'.format(df))
            sys.stderr.flush()
            sys.exit(1)

    # X_train = np.load(open('X_train.npy', 'rb'))
    # y_train = np.load(open('y_train.npy', 'rb'))
    # X_test = np.load(open('X_test.npy', 'rb'))
    # y_test = np.load(open('y_test.npy', 'rb'))

    # local setup for use in PyCharm
    X_train = np.load(open(r'C:\Users\rober\OneDrive\Side Projects\Programming\Machine Learning\Neural_Networks\X_train.npy', 'rb'))
    y_train = np.load(open(r'C:\Users\rober\OneDrive\Side Projects\Programming\Machine Learning\Neural_Networks\y_train.npy', 'rb'))
    X_test = np.load(open(r'C:\Users\rober\OneDrive\Side Projects\Programming\Machine Learning\Neural_Networks\X_test.npy', 'rb'))
    y_test = np.load(open(r'C:\Users\rober\OneDrive\Side Projects\Programming\Machine Learning\Neural_Networks\y_test.npy', 'rb'))

    return X_train, y_train, X_test, y_test


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forwardprop_testcase():
    np.random.seed(2)

    X = np.random.randn(5,3)
    

    W1 = np.array([[0.00416758, -0.00056267, -0.02136196],
       [ 0.01640271, -0.01793436, -0.00841747],
       [ 0.00502881, -0.01245288, -0.01057952],
       [-0.00909008,  0.00551454,  0.02292208]])
     
   
    
    b1 = np.array([[0.,0.,0.,0.]])
    
    W2 = np.array([[ 0.00041539, -0.01117925,  0.00539058, -0.0059616]])
    
    b2 = np.array([[0.]])

    
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    
    
    return X, params


def backprop_testcase():
    np.random.seed(2)

    y = np.random.randint(2, size=(5,1))
    X, params = forwardprop_testcase()
    
    

    z1 = np.array([[ 0.04392812,  0.01215452,  0.02120482, -0.04548798],
       [ 0.02582645,  0.06615439,  0.03948733, -0.04409477],
       [ 0.02539642,  0.03948734,  0.02922894, -0.03568889],
       [-0.0530647,  -0.04409476, -0.03568888,  0.06384614],
       [-0.0107132,   0.01619313,  0.00842731,  0.0058139]])
    a1 = np.array([[ 0.04389989,  0.01215392,  0.02120164, -0.04545663],
       [ 0.02582071,  0.06605805,  0.03946682, -0.04406621],
       [ 0.02539096,  0.03946683, 0.02922062, -0.03567374],
       [-0.05301495, -0.0440662,  -0.03567373,  0.06375953],
       [-0.01071279,  0.01619171,  0.00842711,  0.00581383]])
    z2 = np.array([[2.67647271e-04],[-2.52299642e-04],[-6.04737405e-05],[-1.01805757e-04],[-1.74693941e-04]])
    a2 = np.array([[0.50006691],[0.49993693],[0.49998488],[0.49997455],[0.49995633]])

    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    return X, y, params, cache


def update_params_testcase():
    _, params = forwardprop_testcase()
    

    dW2 = np.array([[ -0.0071034,  -0.01212525, -0.00747344,  0.01038535]])
    db2 = np.array([[0.099983927]])
    dW1 = np.array([[-0.00014218,  0.00010023,  0.00010762],
       [ 0.00382091, -0.00268948, -0.00288968],
       [-0.00184535,  0.00130013,  0.00139629],
       [ 0.00203841, -0.00143624, -0.001542]])
    db1 = np.array([[ 4.13852251e-05, -1.12173654e-03,  5.39304763e-04, -5.94305036e-04]])

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    return params, grads


def nn_model_testcase():
    X, y, _, _ = backprop_testcase()
    return X, y


def linear_forward_testcase():
    np.random.seed(2)
    
    a = np.random.randn(3, 2)
    W = np.random.randn(5, 3)*0.01
    b = np.zeros((5, 1))

    return a, W, b


def activation_forwad_testcase():
    z = np.array([1,-1,0])
    return z


def forward_testcase():
    a, W, b = linear_forward_testcase()
    return a, W, b


def forward_deep_testcase():
    np.random.seed(2)

    X = np.random.randn(10, 100)
    W1 = np.random.randn(200, 10)*0.01
    b1 = np.zeros((200, 1))
    W2 = np.random.randn(100, 200)*0.01
    b2 = np.zeros((100, 1)) 
    W3 = np.random.randn(50, 100)*0.01
    b3 = np.zeros((50, 1)) 
    W4 = np.random.randn(10, 50)*0.01
    b4 = np.zeros((10, 1))
    W5 = np.random.randn(1, 10)*0.01
    b5 = np.zeros((1, 1))

    params = {
        'W1': W1, 'b1': b1,\
        'W2': W2, 'b2': b2,\
        'W3': W3, 'b3': b3,\
        'W4': W4, 'b4': b4,\
        'W5': W5, 'b5': b5,
    }

    return X, params


def activation_backward_testcase():
    np.random.seed(2)

    da = np.random.randn(4,3)
    activation_cache = np.random.randn(4,3)

    return da, activation_cache


def linear_backward_testcase():
    np.random.seed(2)

    dz = np.random.randn(4,3)
    a_prev = np.random.randn(2,3)
    W = np.random.randn(4,2)
    b = np.random.randn(4,1)
    linear_cache = (a_prev, W, b)

    return dz, linear_cache


def backward_testcase():
    da, activation_cache = activation_backward_testcase()
    dz, linear_cache = linear_backward_testcase()
    cache = (linear_cache, activation_cache)

    return da, cache


def backward_deep_testcase():
    """
    X = np.random.rand(3,2)
    y = np.array([[1, 1]])
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}
    aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
           [ 0.02738759,  0.67046751],
           [ 0.4173048 ,  0.55868983]]),
    np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
    np.array([[ 0.]])),
   np.array([[ 0.41791293,  1.91720367]]))])
   """
    np.random.seed(3)
    aL = np.random.randn(1, 2)
    y = np.array([[1, 0]])

    a0 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    z1 = np.random.randn(3,2)
    cache1 = ((a0, W1, b1), z1)

    a1 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    z2 = np.random.randn(1,2)
    cache2 = ( (a1, W2, b2), z2)

    caches = (cache1, cache2)

    return aL, y, caches