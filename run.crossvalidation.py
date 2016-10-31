"""run.py
Script to build model and generate submission file
Performs following operations
    1. Load training data
    2. Preprocess data
    3. Train and validate model using cross validation
    4. Predict on test data
    5. Generate Kaggle submission file
"""

### Importing libraries
import numpy as np
from implementations import *
from model import *
from helpers import *
from proj1_helpers import *
from cross_validation import cross_validation
from prepare_data import prepare_data
import os.path

if __name__ == "__main__":

    ## ========= Load training and test data ========== ##
    print("Loading training data...")
    DATA_TRAIN_PATH = 'train.csv'

    # caching matrix
    if not os.path.isfile("tx_train_edited.npy"):
        y_train, raw_tx_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
        ## ================================================ ##


        ## ========= Preprocess data =============================== ##
        print("Preprocessing data...")
        features_list = [[i,j] for i in range(13) for j in range(i+1,13)]
        tx_train_edited, _ = prepare_data(raw_tx_train, y_train, features_list)
        ## ========================================================= ##
        np.save("y_train", y_train)
        np.save("tx_train_edited", tx_train_edited)
        np.save("raw_tx_train", raw_tx_train)
    else:
        print("Preprocessing data (from cache)...")
        y_train = np.load("y_train.npy")
        tx_train_edited = np.load("tx_train_edited.npy")
        tx_test_edited = np.load("tx_test_edited.npy")
        raw_tx_train = np.load("raw_tx_train.npy")



    ## ========= Train and validate model using cross validation ========== ##
    model = 'least_squares'   ## 'least_squares','ridge_regression','logistic_regression','reg_logistic_regression'
    K = 10                           ## K fold cross validation
    gamma = 0.01                    ## Learning rate for gradient descent
    lambda_ = 0.01                  ## Regularisation parameter
    max_iters = 10                  ## Maximum number of iteration
    initial_w = np.zeros(tx_train_edited.shape[1])# Weight initialisation
    print("Cross validation with model: ", model)
    # Invoke the cross validation function with the above model
    w, loss, avg_val_err, avg_train_err = cross_validation(y_train, tx_train_edited, K, model, lambda_, gamma,
                                                           max_iters, initial_w, num_epochs=1, shuffle=True, seed=1)
    ## ==================================================================== ##


    print("===================")
    print("Training Accuracy: ")
    print((1-avg_train_err)*100)
    print("Validation Accuracy: ")
    print((1-avg_val_err)*100)
    print("===================")




