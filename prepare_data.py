"""
prepare_data.py
Feature processing 
"""

from model import *

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