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
	DATA_TEST_PATH = 'test.csv'

	_, raw_tx_test, ids_test = load_csv_data(DATA_TEST_PATH)
	y_train, raw_tx_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
	## ================================================ ##


	## ========= Preprocess data =============================== ##
	print("Preprocessing data...")
	tx_train_edited, tx_test_edited = prepare_data(raw_tx_train, y_train, raw_tx_test)
	## ========================================================= ##


	## ========= Train and validate model using cross validation ========== ##
	model = 'logistic_regression'   ## 'least_squares','ridge_regression','logistic_regression','reg_logistic_regression'
	K = 2                           ## K fold cross validation
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


    ## ============ Predict on test data ================== ##
	print("Predicting test data...")
	if model=='logistic_regression' or model=='reg_logistic_regression':
		y_pred = np.empty((len(tx_test_edited), 1))

		s = tx_test_edited.dot(w)
		s = sigmoid(tx_test_edited.dot(w))

		y_pred[np.where(s <= .5)] = -1
		y_pred[np.where(s > .5)] = 1
		y_pred = y_pred.reshape(len(tx_test_edited))

	else:
		y_pred = predict_labels(w, tx_test_edited)
    ## ===================================================== ##


    ## === Generate Kaggle submission file ========== ##
	OUTPUT_PATH = 'result.csv'
	create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    ## ============================================== ##



