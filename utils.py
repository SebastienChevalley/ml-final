import numpy as np
import math
nph = np.hstack

def npo(n):
    return np.ones((n,1))

def compute_error(y, x, w):
    return y - x.dot(w)

def compute_mse(e):
    return 1/2*np.mean(e**2)

def ii(cond):
    '''Return the indices assuming condition'''
    return set(np.where(cond)[0])

#Creates polynomial basis
def buildpolyphi(x, features_tuple, indices_to_select, deg):
    n = len(features_tuple)
    phi = npo(len(indices_to_select))
    for feature in features_tuple:
        without_degree_zero = np.vander(x[indices_to_select, feature], deg + 1, increasing=True)[:,1:]
        phi = nph((phi, without_degree_zero))
    return phi

def mesh_indice(sample, feature_min_value, feature_max_value, mesh_resolution):
    feature_range = feature_max_value - feature_min_value

    return int(math.floor((sample - feature_min_value)/ feature_range * (mesh_resolution - 1)))

