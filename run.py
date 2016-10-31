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

if __name__ == "__main__":
    ## ========= Load training and test data ========== ##
    print("Loading training data...")
    DATA_TRAIN_PATH = 'train.csv'
    print("Loading test data...")
    DATA_TEST_PATH = 'test.csv'

    _, raw_tx_test, ids_test = load_csv_data(DATA_TEST_PATH)
    y_train, raw_tx_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
    ## ================================================ ##


    ## ========= Preprocess data =============================== ##
    features_tuple_list = [[i, j] for i in range(13) for j in range(i + 1, 13)]
    tx_train_edited, tx_test_edited = prepare_data(raw_tx_train, y_train, features_tuple_list, raw_tx_test)
    ## ========================================================= ##

    ## ========= Train and validate model using cross validation ========== ##
    model = 'logistic_regression'  ## 'least_squares','ridge_regression','logistic_regression','reg_logistic_regression'
    gamma = 0.01  ## Learning rate for gradient descent
    lambda_ = 0.01  ## Regularisation parameter
    max_iters = 10  ## Maximum number of iteration
    initial_w = np.zeros(tx_train_edited.shape[1])  # Weight initialisation

    print("Train : Logistic regression...")
    w, _ = logistic_regression((y_train + 1) / 2, tx_train_edited, initial_w, max_iters, gamma, seed=1)
    y_pred_train = np.empty((len(tx_train_edited), 1))

    s = sigmoid(tx_train_edited.dot(w))

    y_pred_train[np.where(s <= .5)] = -1
    y_pred_train[np.where(s > .5)] = 1
    y_pred_train = y_pred_train.reshape(len(tx_train_edited))

    print("Score : ", len(ii(y_pred_train == y_train)) / len(y_train))

    print('Test : Predicting')
    y_pred_test = np.empty((len(tx_test_edited), 1))

    s = sigmoid(tx_test_edited.dot(w))

    y_pred_test[np.where(s <= .5)] = -1
    y_pred_test[np.where(s > .5)] = 1
    y_pred_test = y_pred_test.reshape(len(tx_test_edited))

    ## ===================================================== ##


    ## === Generate Kaggle submission file ========== ##
    print("Test : Submiting result.csv")
    OUTPUT_PATH = 'result.csv'
    create_csv_submission(ids_test, y_pred_test, OUTPUT_PATH)
    ## ============================================== ##
