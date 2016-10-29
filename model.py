import numpy as np

from implementations import *

from helpers import *
from utils import *

def partition_dataset(x, y):
    N = x.shape[0]
    Nd = x.shape[1]

    #all indices list
    all_indices = set([j for j in range(N)])
    all_indices_list = list(all_indices)

    #signal indices and background indices lists
    signal_indices = ii(y == 1. )
    background_indices = ii(y == -1.)

    signal_indices_list, background_indices_list = list(signal_indices), list(background_indices)
    signal_indices_list.sort()
    background_indices_list.sort()

    #Jets number indices (0,1,2 or 3)
    jet_values = list(range(4))
    jet_number_indices = [ii(x[:, 22] == jet_number_value) for jet_number_value in jet_values]
    jet_number_indices_list = [sorted(list(x)) for x in jet_number_indices]

    #Undefined values indices lists
    undefined_values_indices = [None]*Nd
    defined_values_indices = [None]*Nd
    undefined_values_indices_list = [None]*Nd
    defined_values_indices_list = [None]*Nd

    for j in range(Nd):
        undefined_values_indices[j] = ii(x[:,j] == -999.)
        defined_values_indices[j] = ii(x[:,j] != -999.)

        undefined_values_indices_list[j] = list(undefined_values_indices[j])
        defined_values_indices_list[j] = list(defined_values_indices[j])

        undefined_values_indices_list[j].sort()
        defined_values_indices_list[j].sort()

    # partitionning according to feature 22 value and samples defined or not for the feature 0
    # it happens that 0th feature has undefined value without respect to value of 22th feature
    partitions = [jet_number_indices[i] & defined_values_indices[0] for i in range(len(jet_values))] \
                 + [jet_number_indices[i] & undefined_values_indices[0] for i in range(len(jet_values))]

    return partitions, all_indices, defined_values_indices

def trainprocess(x, y):
    ## ============================================= ##
    ## ========= Preprocessing parameters ========== ##
    ## ============================================= ##

    # for each tuple of features, an approximation of the probability density will be computed (normalized from -1 to 1)
    features_tuple_list = [[i] for i in range(13)]

    #degree for each feature to be polynomialized
    degree = 8

    #there are less points in proportition for a tuple of features
    ignoring_threshold = 5e-5

    #number of blocks for each feature to be considered
    resolution = 200

    ## ============================================= ##
    ## ============= Features creation ============= ##
    ## ============================================= ##

    #the new features
    newX = []

    #intermediate weights for each new feature (coeffecients of the polyomial approximation)
    intermediate_weights = []

    #windows to be considered (outside <=> excluded <=> probability density = 0)
    intermediate_windows = []

    # if a tuple of original features according a subset partition has no samples, it is excluded (<=> False)
    is_features_included = []

    partitions, all_indices, defined_values_indices = partition_dataset(x, y)

    for partition in partitions:
        for features_tuple in features_tuple_list:

            indices_to_select = all_indices.copy() & partition.copy() #indices to be considered

            for feature in features_tuple:
                indices_to_select &= defined_values_indices[feature] # only consider correct samples (not -999 values)

            indices_to_select_list = list(indices_to_select)

            if len(indices_to_select_list) == 0:
                print('u',features_tuple)
                is_features_included.append(False)
                continue
                #Create the mesh grid for the tuple of features

            features_window = []
            features_min = np.min(x[indices_to_select_list], axis=0)
            features_max = np.max(x[indices_to_select_list], axis=0)

            meshed_features= [np.linspace(features_min[feature], features_max[feature], resolution)
                              for feature in features_tuple]
            features_mesh_grid = np.meshgrid(*meshed_features, indexing='ij')

            # Create probability density for the tuple of features in the meshgrid
            meshed_density = np.zeros(features_mesh_grid[0].shape)
            meshed_sample_number = np.zeros(features_mesh_grid[0].shape)

            # for each sample
            for i in indices_to_select:
                mesh_indices = tuple([mesh_indice(x[i, feature], features_min[feature], features_max[feature], resolution)
                                      for feature in features_tuple])

                meshed_density[mesh_indices] += y[i]
                meshed_sample_number[mesh_indices] += 1

            meshed_cell_not_empty_indices = np.where(meshed_sample_number !=0 )
            meshed_density[meshed_cell_not_empty_indices] /= meshed_sample_number[meshed_cell_not_empty_indices] #to have a density from -1 to 1

            #We 'undo' the meshgrid and give the probability to each sample
            #Thus it gives new features to newX

            newX.append(np.zeros(len(y)))
            indices_to_select_filtered = []
            for i in indices_to_select:
                mesh_indices = tuple([mesh_indice(x[i, n], features_min[n], features_max[n], resolution)
                                      for n in features_tuple])

                newX[-1][i] = meshed_density[mesh_indices]

                #select only if there is enough samples concentrated on mesh values
                if meshed_sample_number[mesh_indices] >= ignoring_threshold * len(indices_to_select):
                    indices_to_select_filtered.append(i)

            include_feature = len(indices_to_select_filtered) != 0
            is_features_included.append(include_feature)

            w = 0

            if include_feature:
                intermediate_phi = buildpolyphi(x.copy(), features_tuple, indices_to_select_filtered, degree)
                loss, w = least_squares(newX[-1][indices_to_select_filtered], intermediate_phi)
                print(features_tuple, ":", loss)

                intermediate_weights.append(w)
                intermediate_windows.append([(np.min(feature), np.max(feature))
                                             for feature in features_tuple_list])

    return intermediate_weights, intermediate_windows, is_features_included


def predictprocess(x, y, intermediate_weights, intermediate_windows, acks):
    ## ============================================= ##
    ## ========= Preprocessing parameters ========== ##
    ## ============================================= ##

    # for each tuple of features, an approximation of the probability density will be computed (normalized from -1 to 1)
    features_tuple_list = [[i] for i in range(13)]

    #degree for each feature to be polynomialized
    degree = 8

    ## ============================================= ##
    ## ============= Features creation ============= ##
    ## ============================================= ##

    N = x.shape[0]
    partitions, all_indices, defined_values_indices = partition_dataset(x, y)

    newX = [] #the new features
    j=0
    for partition in partitions:
        for ack, featuresTuple in zip(acks, features_tuple_list):
            if not ack:
                continue

            selected_indices = all_indices.copy() & partition.copy() #indices to be considered
            for n, feature in enumerate(featuresTuple):
                sample = x[:,feature]
                feature_min, feature_max = intermediate_windows[j][n]

                # only consider correct samples (not -999 values
                selected_indices &= defined_values_indices[feature] \
                                    & ii(sample > feature_min) \
                                    & ii(sample < feature_max)

            selected_indices_list = list(selected_indices)

            intermediate_phi = np.zeros((N, degree + 1))
            intermediate_phi[selected_indices_list] = buildpolyphi(
                x.copy(),
                featuresTuple,
                selected_indices_list,
                degree
            )

            newX.append(intermediate_phi.dot(intermediate_weights[j]))
            j += 1

    newX = np.array(newX[:][:]).reshape(len(newX),N).T

    return newX
