import numpy as np

from implementations import *
from model import *
from helpers import *
from proj1_helpers import *

from cross_validation import cross_validation

DATA_TRAIN_PATH = '../../my_work/train.csv'
DATA_TEST_PATH = '../../my_work/test.csv'
OUTPUT_PATH = 'result.csv'

print("Load dataset...")

_, raw_tx_test, ids_test = load_csv_data(DATA_TEST_PATH)
y_train, raw_tx_train, ids_train = load_csv_data(DATA_TRAIN_PATH)


def prepare_data(raw_tx_train, y_train, raw_tx_test= None):
    """
    Load an prepare data matrix,
    returns the tx matrix for training and testing set (if provided)
    """
    print("Building new dataset (1/3)...")
    intermediate_weights, intermediate_windows, ack = trainprocess(raw_tx_train, y_train)
    print("Building new dataset (2/3)...")
    tx_train_edited = predictprocess(raw_tx_train, intermediate_weights, intermediate_windows, ack)

    tx_test_edited = None
    if raw_tx_test is not None:
        print("Building new dataset (3/3)...")
        tx_test_edited = predictprocess(raw_tx_test, intermediate_weights, intermediate_windows, ack)
    else:
        print("Building new dataset (3/3) [skipped]...")

    return tx_train_edited, tx_test_edited


def run_prediction(raw_tx_train,
                   y_train,
                   raw_tx_test,
                   ids_test,
                   method= logistic_regression,
                   output_path= OUTPUT_PATH):

    tx_train_edited, tx_test_edited = prepare_data(raw_tx_train, y_train, raw_tx_test)

    print("Training process...")
    weights_train, loss_train = method(
        (y_train + 1) / 2,
        tx_train_edited,
        np.zeros((tx_train_edited.shape[1], 1)),
        3,
        .01
    )

    print("Predict process...")
    y_pred = np.empty((len(tx_test_edited), 1))

    s = tx_test_edited.dot(weights_train)
    if method == logistic_regression:
        print("apply sigmoid")
        s = sigmoid(tx_test_edited.dot(weights_train))

    y_pred[np.where(s <= .5)] = -1
    y_pred[np.where(s > .5)] = 1
    y_pred = y_pred.reshape(len(tx_test_edited))

    #y_pred = predict_labels(wtrain2, phi1)
    create_csv_submission(ids_test, y_pred, output_path)

def run_cross_validation(tx, y):
    tx_edited, _ = prepare_data(tx, y)

    # TODO cross_validation(y, tx_edited, 10, "least_squares_SGD", )

#run_prediction(raw_tx_train, y_train, raw_tx_test, ids_test)
run_cross_validation(raw_tx_train, y_train)