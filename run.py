import numpy as np
from implementations import *
from model import *
from helpers import *
from proj1_helpers import *

DATA_TRAIN_PATH = '../../my_work/train.csv' # TODO: download train data and supply path here
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

x = tX.copy()
intermediate_weights, intermediate_windows, ack = trainprocess(x, y)

xx = predictprocess(x, y, intermediate_weights, intermediate_windows, ack)

wtrain2,l1 = logistic_regression((y + 1) / 2, xx, np.zeros((xx.shape[1], 1)), 3, .01)

y_pred = np.empty((len(xx),1))
s = sigmoid(xx.dot(wtrain2))
y_pred[np.where(s <= .5)] = -1
y_pred[np.where(s > .5)] = 1
y_pred = y_pred.reshape(len(xx))

OUTPUT_PATH = 'result.csv' # TODO: fill in desired name of output file for submission
#y_pred = predict_labels(wtrain2, phi1)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)