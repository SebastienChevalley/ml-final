import numpy as np

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

from implementations import calculate_mse, logistic_regression
from implementations import ridge_regression
from implementations import build_poly
from implementations import least_squares

def cross_validation(y, x, k_indices, k, degree= 1,
                     lambda_= None, gamma= None,
                     learn_method= lambda y, tx, lambda_= None, gamma= None: ridge_regression(y, tx, lambda_),
                     compute_error= calculate_mse):

    """return the loss of ridge regression."""
    # ***************************************************
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************

    test_raw_x = x[k_indices[k]]
    test_y = y[k_indices[k]]

    k_train_indices = np.delete(k_indices, k, axis=0).flatten()

    train_raw_x = x[k_train_indices]
    train_y = y[k_train_indices]

    # ***************************************************
    # form data with polynomial degree: TODO
    # ***************************************************

    test_x = build_poly(test_raw_x, degree)
    train_x = build_poly(train_raw_x, degree)

    # ***************************************************
    # ridge regression: TODO
    # ***************************************************

    #print([x.shape for x in [train_x, np.matrix(train_y), test_x, test_y]])

    mse, w = learn_method(train_y, train_x, lambda_= lambda_, gamma= gamma)
    loss_te = np.asscalar(compute_error(test_y, test_x, w))
    loss_tr = np.asscalar(mse)

    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************

    return loss_tr, loss_te

from plots import cross_validation_visualization

def cross_validation_demo(tx,
                          y,
                          learn_method= lambda y, tx, lambda_= None, gamma= None: logistic_regression(y, tx, np.zeros((tx.shape[1],)), 100, lambda_),
                          compute_error= calculate_mse):
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 2, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # cross validation: TODO
    # ***************************************************

    gamma = 0
    for lambda_ in lambdas:
        train_error = 0
        test_error = 0
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k,
                                                degree= degree,
                                                lambda_= lambda_,
                                                gamma= gamma,
                                                learn_method= learn_method,
                                                compute_error= compute_error)
            train_error += loss_tr
            test_error += loss_te
        rmse_tr.append(np.sqrt(2 * train_error / k_fold))
        rmse_te.append(np.sqrt(2 * test_error / k_fold))


    cross_validation_visualization(lambdas, rmse_tr, rmse_te)


