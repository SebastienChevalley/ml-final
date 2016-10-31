"""
implementations.py
Implements 'least_squares','least_squares_GD','least_squares_SGD','ridge_regression',
            'logistic_regression','reg_logistic_regression'
"""

"""
toolbox.py
Implements all the helper functions 
"""

import numpy as np
from helpers import *

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_loss(y, tx, w):
    """Calculate the loss.
    using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)

def standardize_m(x):
    """Standardises data"""
    x = (x-np.mean(x,axis=0))/(np.std(x,axis=0)+1e-6)
    return x

def build_poly(x, degree):
    """polynomial basis function."""
    return np.c_[[x**i for i in range(1,degree+1)]].transpose()    

def compute_gradient(y, tx, w):
    """Computes the gradient"""
    return -np.dot(tx.transpose(), y-np.dot(tx, w))/len(y)

def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        delL = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * delL
    return loss, w

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    return -np.dot(tx.transpose(),y-np.dot(tx,w))/len(y)

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        for n_iter in range(max_iters):
            delL = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            w = w - gamma*delL
    return loss, w


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Does one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    # compute the cost
    loss = calculate_loss(y, tx, w)
    # compute the gradient
    gradient = calculate_gradient(y, tx, w)
    # update w
    w = w - gamma * gradient
    return loss, w

def sigmoid(t):
    """apply sigmoid function on t."""
    t = np.array(t)
    ## Handling overflow/underflow
    plus = np.where(t>=0)
    minus = np.where(t<0)
    sigmd = t #np.empty(len(t)).reshape((len(t),1))
    sigmd[minus] = np.exp(t[minus])/(1+np.exp(t[minus]))
    sigmd[plus] = 1/(1+np.exp(-t[plus]))
    return sigmd

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    act = np.dot(tx,w)  ## Activation
    #act = np.array(act)
    ## Handling overflow/underflow
    plus = np.where(act>100)
    minus = np.where(act<=100)
    logexp = act #np.empty(len(act)) #.reshape((len(act),1))
    logexp[plus] = act[plus]
    logexp[minus] = np.log(1+np.exp(act[minus]))
    loss = sum(logexp) - np.dot(y,act)
    return loss
    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    tmp1 = sigmoid(np.dot(tx,w))
    tmp2 = (tmp1 - y)
    return np.dot(tx.transpose(),tmp2)

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # return loss, gradient
    loss = calculate_loss(y, tx, w) + lambda_*sum(w**2)
    gradient = calculate_gradient(y, tx, w) + 2*lambda_*(sum(w))
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Does one step of gradient descent, using the penalized logistic regression.
    Returns the loss and updated w.
    """
    # return loss, gradient
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    # update w
    w = w - gamma * gradient
    return loss, w 


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """calculates the least squares solution using gradient descent."""
    loss, w = gradient_descent(y, tx, initial_w, max_iters, gamma)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """calculates the least squares solution using stochastic gradient descent."""
    batch_size = 1  ## Batch size of 1
    loss, w = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma)
    return w, loss

def least_squares(y, tx):
    """calculates the least squares solution using normal equation."""
    w = np.dot(np.linalg.inv(np.dot(tx.transpose(), tx)), np.dot(tx.transpose(), y))
    loss = sum((y-np.dot(tx,w))**2)/(2*len(y))
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implements ridge regression using normal equation"""
    lambdaI = lambda_*np.eye(tx.shape[1])
    w = np.dot(np.linalg.inv(np.dot(tx.transpose(),tx) + np.dot(lambdaI.transpose(),lambdaI)),np.dot(tx.transpose(),y))
    loss = sum((y-np.dot(tx,w))**2)/(2*len(y))
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma, seed= None):
    """Implements logistic regression using stochastic gradient descent"""
    batch_size = 1  ## Batch size of 1
    w = initial_w   ## Inital weight
    # Iterate over each training sample
    loss = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, seed= seed):
        for iter in range(max_iters):
            # Loop over max_iters
            # computes loss and updates w using gradient
            loss, w = learning_by_gradient_descent(minibatch_y, minibatch_tx, w, gamma)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Implements regularised logistic regression using stochastic gradient descent"""
    batch_size = 1  ## Batch size of 1
    w = initial_w   ## Inital weight
    # Iterate over each training sample
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        for iter in range(max_iters): 
            # Loop over max_iters
            # computes loss and updates w using gradient
            loss, w = learning_by_penalized_gradient(minibatch_y, minibatch_tx, w, gamma, lambda_)
    return w, loss



