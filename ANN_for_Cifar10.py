# import libraries
import numpy as np
import tensorflow as tf
import keras
import scipy.stats
import matplotlib.pyplot as plt
import pickle
import glob
import time
from random import random
from collections import defaultdict
from skimage.transform import rescale, resize
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband


# Load images
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# Change image size to given scale
def cifar10_color(data, im_rescale):
    """
    Convert an image of(n , 32, 32,3) into given size and flatten them.

    :param
    data(ndarray): Image data

    :return(ndarray):
    Resized image  size
    """

    resized_img_array = np.zeros((data.shape[0], im_rescale * im_rescale * 3))  # store resized image array
    for i in range(data.shape[0]):  # loop through each image
        img_1x1 = resize(data[i], (im_rescale, im_rescale))  # resize (32,32) to (resize,resize)
        resized_img_array[i, :] = img_1x1.flatten()

    return resized_img_array


# Define keras-tuner HyperModel subclass for model building
class Cifar_10_hyper_tunner(HyperModel):

    def __init__(self, num_class, input_shape):
        self.num_class = num_class
        self.input_shape = input_shape

    # Model build function
    def build(self, hp):
        model = Sequential()
        model.add(layers.Flatten(input_shape=self.input_shape))

        for layer in range(hp.Int('num_layers', 1, 4)):  # Layers
            model.add(layers.Dense(units=hp.Int('units_' + str(layer),
                                                min_value=5,
                                                max_value=500,
                                                step=100,
                                                default=128),
                                   activation=hp.Choice("dense_activation",
                                                        values=["relu", "tanh", "sigmoid"],
                                                        default="relu")))
            model.add(layers.BatchNormalization())  # Batch normalization

        model.add(layers.Dense(self.num_class, activation='softmax'))  # Output layer
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.RMSprop(
                          hp.Float(
                              "learning_rate",
                              min_value=1e-4,
                              max_value=1e-2,
                              sampling="log",
                              default=1e-3)),  # Learning rate
                      metrics=['accuracy'])  # Compile with  configuration

        return model



if __name__ == "__main__":

    np.random.seed(100)  # seed random generator

    # Test data
    test_datadict = unpickle('cifar-10-batches/test_batch')  # change file path
    test_data = test_datadict['data']  # test data
    # test_data = test_data.astype(np.int16)  # change data type
    test_data = test_data.reshape(test_data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype(
        "uint8")  # reshape to orignal image size
    test_labels = np.array(test_datadict['labels'], dtype=np.int16)  # test labels
    print('Test data shape:  {}'.format(test_data.shape))
    print('Test Labels shape:  {}\n'.format(test_labels.shape))

    # Train dataset
    data_batchs = glob.glob('cifar-10-batches/data_batch*')  # list of data_batches' paths
    train_data = np.empty(shape=[0, 3072], dtype=np.int16)  # empty array for train data
    train_labels = np.empty(shape=[0], dtype=np.int16)  # empty array for train labels

    # Concatenate all Train data_batches
    for batch in data_batchs:
        train_datadict = unpickle(batch)  # unpickle file
        data_batch = train_datadict['data']  # batch data
        train_data = np.vstack((train_data, data_batch))  # stack batches data

        batch_labels = np.array(train_datadict['labels'])  # batch labels
        train_labels = np.hstack((train_labels, batch_labels))  # stack batches labels

    # reshape to orginal image size
    train_data = train_data.reshape(train_data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    print('Train data shape:  {}'.format(train_data.shape))
    print('Train Labels shape:  {}\n'.format(train_labels.shape))

    # Display random some images
    labeldict = unpickle('cifar-10-batches/batches.meta')  # change file path
    label_names = labeldict["label_names"]
    # Show some images randomly
    for i in range(test_data.shape[0]):

        if random() > 0.999:
            img_1x1 = resize(test_data[i], (8, 8))  # Convert images to mean values of each color channel
            plt.figure(1);
            plt.clf()
            plt.imshow(img_1x1)
            plt.title(f"Image {i} label={label_names[test_labels[i]]} (num {test_labels[i]})")
            plt.pause(1)

    # Scale image with min-max scaler
    train_data = train_data / 255.0
    test_data = test_data / 255.0
    train_labels_cat = to_categorical(train_labels)  # one-hot encoding
    test_labels_cat = to_categorical(test_labels)

    # RandomSearch Hyper-Parameter Tunner
    input_shape = (32, 32, 3)  # Image shape
    num_class = 10  # Total classes
    hype_tunner = Cifar_10_hyper_tunner(num_class=num_class,
                                       input_shape=input_shape)  # Create hyper-model class instance
    # RandomSearch hyper-parameters Search
    model_tuner = RandomSearch(hype_tunner,
                               objective='val_accuracy',
                               max_trials=20,  # Total no. of trails to perform
                               executions_per_trial=2,  # No. of test in a trail (reduce results variance)
                               directory='RandomSearch',
                               project_name='Cifar_10')
    model_tuner.search_space_summary() # Show a summary of the search

    # Fit train data for Hyper-parameters search
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')  # Early stopping
    model_tuner.search(train_data,
                       train_labels_cat,
                       epochs =100,
                       shuffle = True,
                       batch_size=64,
                       validation_split=0.2,
                       callbacks=[early_stop],
                       verbose=0) # Parameter search

    # Evaluate test data with the best model
    best_model = model_tuner.get_best_models(num_models=1)[0]  # best model
    loss, accuracy = best_model.evaluate(test_data, test_labels_cat)  # Evaluation
    print(accuracy)

    # Plot Model Comparison
    accuracy = [45.46, 43.46, 51.46]  # Y-values # Y-values (Accuracy score from each model types)
    x = ['1NN(L1+Standarised)', 'Multi-Gau_Bayes(16x16)', 'ANN']  # X-values
    fig, ax = plt.subplots(figsize=(9, 7))  # Set figure
    bar_plot = plt.bar(x, accuracy, width=0.2, color='maroon')  # Barchart plot
    for x in bar_plot:  # Get (x,y) co-ordinates
        height = x.get_height()  # Get bar values
        ax.text(x=x.get_x() + x.get_width() / 2., y=height, s=str(height) + '%', ha='center', va='bottom',
                rotation=0)  # Put values
    plt.xlabel('Model Types')
    plt.ylabel('Accuracy')
    plt.title('Comparision: 1NN vs Bayes vs ANN')
    plt.ylim(0, 90)
    plt.show()
