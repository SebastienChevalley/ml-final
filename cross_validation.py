"""
cross_validation.py
Performs K fold cross validation
Inputs:
    1. y: Labels
    2. x: Features
    3. K: K fold
    4. model: 'least_squares','ridge_regression','logistic_regression','reg_logistic_regression'
    5. max_iters: Maximum number of iterations
    6. initial_w: Weight initialisation
    7. num_epochs: Number of epochs (default set to 1)
    8. shuffle: Shuffle input data for each epoch (default: True)
    9. seed: Random seed (default: 1)
Outputs:
    1. w : Optimal weight
    2. loss : Loss metric
    3. avg_val_err: Average validation error
    4. avg_train_err: Average training error
"""
from implementations import *
from proj1_helpers import *
import numpy as np

## Cross validation
def cross_validation(y, x, K, model, lambda_, gamma, max_iters, initial_w, num_epochs=1, shuffle=True, seed=1):
    data_size = len(y)   ## Number of data points
    x = np.array(x)
    y = np.array(y)

    count = 0
    batch_size = int(data_size/K) ## Data size assumed to multiple of K
    err_val = np.zeros(num_epochs*K)
    err_train = np.zeros(num_epochs*K)

    # Set random seed such that model comparison is uniform and consistent
    np.random.seed(seed)

    # Number of times K-fold cross validation is repeated
    for epoch in range(num_epochs):

        # Randomize to remove ordering in the input data
        if shuffle == True:
            shuffle_ind = np.random.permutation(np.arange(data_size))
            y_shuffle = y[shuffle_ind]
            x_shuffle = x[shuffle_ind]
        else:
            y_shuffle = y
            x_shuffle = x

        # K-fold cross validation
        for k in range(0,K):

            # Select validation data in kth fold
            start_val_ind = k*batch_size
            end_val_ind   = (k+1)*batch_size
            y_val = y_shuffle[start_val_ind: end_val_ind]
            x_val = x_shuffle[start_val_ind: end_val_ind]

            # Select training data in kth fold
            train_ind = np.setxor1d(range(0,data_size),range(start_val_ind,end_val_ind))
            y_train = y_shuffle[train_ind]
            x_train = x_shuffle[train_ind]

            # Logistic regression and reg_logistic_regression models
            if ((model=='logistic_regression') or (model=='reg_logistic_regression')):
                if model=='logistic_regression':    ## Train the model
                    w, loss = logistic_regression(y_train, x_train, initial_w, max_iters, gamma)
                elif model=='reg_logistic_regression':
                    w, loss = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)
                # Predict on validation and training data
                y_pred_val = np.ones(len(y_val))
                y_pred_val[sigmoid(np.dot(x_val,w)) <= 0.5] = -1
                y_pred_train = np.ones(len(y_train))
                y_pred_train[sigmoid(np.dot(x_train,w)) <= 0.5] = -1


                # Least squares models
            else:
                if model == 'least_squares':                    ## Least squares regression using normal equation
                    w, loss = least_squares(y_train, x_train)
                elif model == 'least_squares_GD':               ## Least squares regression using gradient descent
                    w, loss = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma)
                elif model == 'least_squares_SGD':              ## Least squares regression using stochastic gradient descent
                    w, loss = least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma)
                elif model == 'ridge_regression':               ## Ridge regression
                    w, loss = ridge_regression(y_train, x_train, lambda_)
                else:
                    print("Unknown model")
                # Predict on validation and training data
                y_pred_val = predict_labels(w, x_val)                  ## Predict on validation data
                y_pred_train = predict_labels(w, x_train)              ## Predict on training data

            err_val[count] = sum(y_pred_val!=y_val)/len(y_val)           ## Accuaracy on Validation data
            err_train[count] = sum(y_pred_train!=y_train)/len(y_train)   ## Accuaracy on training data
            count+=1

        ## Average error over all folds of cross validation
        avg_val_err = np.mean(err_val)
        avg_train_err = np.mean(err_train)

    ## Return optimal weight, loss, average validation error, average training error
    return w, loss, avg_val_err, avg_train_err

