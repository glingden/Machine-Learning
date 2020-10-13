""""
Author: Ganga Lingden
Tested KNN(K-Nearest Neighbours) with two different ways :
  1. Totally scratch with numpy and python only, and Tested cifar dataset with out data standardized: Results: L1 (38.59%) and L2 (35.39%)
  2. With scipy package and data is standardized using z-score: l1 (45.46%) and l2 ( 42.26%)
Dataset: Cifar-10(Python version), Link: https://www.cs.toronto.edu/~kriz/cifar.html
"""

# Import Packages
import pickle
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
from random import random
from scipy.stats import zscore
from scipy.spatial import distance


# Define function to read file
def unpickle(file):
    with open(file, 'rb') as f:
        dict_val = pickle.load(f, encoding="latin1")
    return dict_val


# Define accuracy function
def class_acc(x, y):
    """
    Find accuracy percentage of given predicted and true labels.
    Parameters
    ----------
    x (ndarray): True label
    y (ndarray): Predicted label
    Return
    ------
    accuracy: Accuracy percentage
    """

    if len(x) == len(y):
        accuracy = (np.sum(x == y) / len(x)) * 100
        return np.round(accuracy, decimals=2)
    else:
        print('Predicted and Ground truth must have same length')


# Define random classifier function
def cifar10_classifier_random(x, label):
    """
    parameters:
    ----------
    x (ndarray): Test dataset
    label(int): No. of labels/Class
    Return:
    --------
    labels (1d array): Predicted labels
    """
    labels = np.array(range(label))

    # Random guess with uniform distribution of each class
    random_labels = np.random.choice(labels, len(x))
    return random_labels


# Define class for calculating distance metric
class DistanceMetric:
    """
    Calculate distance metrics: 1.Euclidean(L2) and 2.Manhattan(L1)
    Parameter:
     x (nd array):  vector/matrix
     y (nd array): vector/matrix

    Return:
     distance (1d array):  Euclidean or Manhattan distance
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Euclidean distance
    def euclidean_distance(self):

        # vector to vector(1d to 1d)
        if self.x.ndim == 1 and self.y.ndim == 1:
            eud_dist = np.sum((self.x - self.x) ** 2)
            return eud_dist

        # matrix to matrix
        elif self.x.ndim > 1 and self.y.ndim > 1:
            if self.x.shape[0] == 1:  # Single data
                eud_dist = np.sum(((self.x - self.y) ** 2), axis=1)
                eud_dist = np.expand_dims(eud_dist, axis=0)  # Change to 2d
                return eud_dist

            else:
                self.x = self.x[:, None]  # Change to 3d
                eud_dist = np.sum(((self.x - self.y) ** 2), axis=2)
                return eud_dist

        # vector to matrix(1d to 2d)
        else:
            eud_dist = np.sum(((self.x - self.y) ** 2), axis=1)
            return eud_dist

    # Manhattan distance
    def manhattan_dist(self):
        # vector to vector(1d to 1d)
        if self.x.ndim == 1 and self.y.ndim == 1:
            man_dist = np.sum(np.abs(self.x - self.y))
            return man_dist

        # matrix to matrix(2d to 2d)
        elif self.x.ndim > 1 and self.y.ndim > 1:
            if self.x.shape[0] == 1:
                man_dist = np.sum(np.abs(self.x - self.y), axis=1)
                man_dist = np.expand_dims(man_dist, axis=0)  # Change to 2d
                return man_dist
            else:
                self.x = self.x[:, None]  # Change to 3d
                man_dist = np.sum(np.abs(self.x - self.y), axis=2)
                return man_dist

        # vector to matrix(1d to 2d)
        else:
            man_dist = np.sum(np.abs(self.x - self.y), axis=1)
            return man_dist


# Define Ecludian distance function
def euclidean_distance(x, y):
    """
    Calculate Euclidean distance

    parameter
    ---------
     x (nd array):  vector/matrix
     y (nd array): vector/matrix

     return:
      eud_dist (1d array):  Euclidean distance

    """

    # use for vector to vector
    if x.ndim == 1 and y.ndim == 1:
        eud_dist = np.sum((x - y) ** 2)
        return eud_dist

    # matrix to matrix
    elif x.ndim > 1 and y.ndim > 1:
        if x.shape[0] == 1:
            eud_dist = np.sum(((x - y) ** 2), axis=1)
            eud_dist = np.expand_dims(eud_dist, axis=0)  # Change to 2d
            return eud_dist
        else:
            x = x[:, None]  # Change to 3d
            eud_dist = np.sum(((x - y) ** 2), axis=2)
            return eud_dist

    # vector to matrix
    else:
        eud_dist = np.sum(((x - y) ** 2), axis=1)
        return eud_dist


# Manhattan distance function
def manhattan_dist(x, y):
    # vector to vector(1d to 1d)
    if x.ndim == 1 and y.ndim == 1:
        man_dist = np.sum(np.abs(x - y))
        return man_dist

    # matrix to matrix(2d to 2d)
    elif x.ndim > 1 and y.ndim > 1:
        if x.shape[0] == 1:
            man_dist = np.sum(np.abs(x - y), axis=1)
            man_dist = np.expand_dims(man_dist, axis=0)  # Change to 2d
            return man_dist
        else:
            x = x[:, None]  # Change to 3d
            man_dist = np.sum(np.abs(x - y), axis=2)
            return man_dist

    # vector to matrix(1d to 2d)
    else:
        man_dist = np.sum(np.abs(x - y), axis=1)
        return man_dist


# Define K-Nearest Neighbors Classifier
class KnnClassifier:
    """K-Nearest Neighbors Classifier"""

    def __init__(self, metric, k=1):
        self.k = k
        self.metric = metric

    # Fit data for training
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = x_train

    # Prediction
    def prediction(self, x_test):

        # Change to 2d
        if x_test.ndim == 1:
            x_test = np.expand_dims(x_test, axis=0)

        dist = self.metric(x_test, self.x_train)  # Distance
        k_index = np.argsort(dist, axis=1)[:,0:self.k]  # index of k-nearest neighbors
        k_neighbors = self.y_train[k_index]  # k-nearest neighbors (2d)

        predicted_label = np.zeros(len(test), dtype=int)  # store prediction classes
        for i, neighbour in enumerate(k_neighbors):
            values, counts = np.unique(neighbour, return_counts=True)  # count each neighbors
            highest_count_neighbor = values[np.argsort(-counts)][0]  # neighbors with highest count
            predicted_label[i] = highest_count_neighbor
        return predicted_label
'''
        else:
            k_index = np.argsort(dist)[:self.k]  # index of k-nearest neighbors
            neighbors = tr_label[k_index]  # k-nearest neighbors
            values, count = np.unique(neighbors, return_counts=True)  # count each neighbors
            highest_count_neighbor = values[np.argsort(-counts)][0]  # neighbors with highest count
            predicted_label[i] = highest_count_neighbor
            return predicted_label
'''

# Define 1NN classifier function
def cifar10_classifier_1nn(x, tr_data, tr_label):
    """
       1 Nearest Neighbour Classifier
       Parameters
       ----------
       x (ndarray) : Input vector
       train_data (ndarray): Training dataset of shape (m*d)
       train_labels (ndarray): Train labels

       Returns
       -------
       predicted_label(ndarray) : Predicted lable of Input Vector

       """

    # Input is vector
    if x.ndim == 1:
        elu_dist = manhattan_dist(x, tr_data)  # euclidean function
        sorted_index = np.argsort(elu_dist)[0]  # first index
        predicted_label = tr_label[sorted_index]  # label with first index
        return predicted_label

    # Input is matrix
    else:
        elu_dist = manhattan_dist(x, tr_data)  # euclidean function
        sorted_index = np.argsort(elu_dist)[:, 0]  # first index
        predicted_label = tr_label[sorted_index]  # label with first index
        return predicted_label


if __name__ == "__main__":

    np.random.seed(100)  # seed random generator

    # Test data
    test_datadict = unpickle('../cifar-10-batches/test_batch')  # change file path
    test_data = test_datadict['data']  # test data
    test_data = test_data.astype(np.int16)  # change data type
    test_labels = np.array(test_datadict['labels'], dtype=np.int16)  # test labels
    print('Test data shape:  {}'.format(test_data.shape))
    print('Test Labels shape:  {}\n'.format(test_labels.shape))

    # Train dataset
    data_batchs = glob.glob('../cifar-10-batches/data_batch*')  # list of data_batches' paths
    train_data = np.empty(shape=[0, 3072], dtype=np.int16)  # empty array for train data
    train_labels = np.empty(shape=[0], dtype=np.int16)  # empty array for train labels

    # Concatenate all Train data_batches
    for batch in data_batchs:
        train_datadict = unpickle(batch)  # unpickle file
        data_batch = train_datadict['data']  # batch data
        train_data = np.vstack((train_data, data_batch))  # stack batches data

        batch_labels = np.array(train_datadict['labels'])  # batch labels
        train_labels = np.hstack((train_labels, batch_labels))  # stack batches labels

    print('Train data shape:  {}'.format(train_data.shape))
    print('Train Labels shape:  {}\n'.format(train_labels.shape))
  
    # Display random some images
    labeldict = unpickle('../cifar-10-batches/batches.meta')  # change file path
    label_names = labeldict["label_names"]

    X_test = test_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y_test = np.array(test_labels)

    for i in range(X_test.shape[0]):
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1);
            plt.clf()
            plt.imshow(X_test[i])
            plt.title(f"Image {i} label={label_names[Y_test[i]]} num ({Y_test[i]}))")
            plt.pause(1)
    
    # Random classifier
    random_labels = cifar10_classifier_random(test_data, 10)
    random_accu = class_acc(test_labels, random_labels)  # accuracy fucntion
    print('Random Classifier Accuracy: {}% \n'.format(random_accu))

    # 1NN classifier
    print('1NN Classifier Running .......  ')
    start_time = time.time()  # start time
    predicted_labels = []  # store predicted lables

    # Loop all test data instances
    for x in test_data:
        pred_label = cifar10_classifier_1nn(x, train_data, train_labels)  # call 1NN classifier funtion
        predicted_labels.append(pred_label)  # append

    accu_1NN = class_acc(test_labels, np.array(predicted_labels))  # accuracy function
    print("classifier accuracy:  {} %".format(accu_1NN))
    print("Time taken:  {} seconds".format(round(time.time() - start_time)))
   
    """"  ====== Using Scipy =======     """
    # Using Scipy for distance calculation and z-score
    test_data = zscore(test_data, axis=1)  # standarized data
    train_data = zscore(train_data, axis=1)
    start_time = time.time()  # start time

    eudis = distance.cdist(test_data, train_data, 'euclidean')  # l2 distance
    index_sorted = np.argsort(eudis)[:, 0]  # pick first closet one
    accu_1NN = class_acc(test_labels, train_labels[index_sorted])  # accuracy function
    print('Accuracy:  {}%'.format(accu_1NN))
    print("Time taken:  {} seconds".format(round(time.time() - start_time)))
