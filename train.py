# =====================================================================================================================================
# === About
# =====================================================================================================================================
# Name:        train.py 
# Author:      AlexVezo
# Date:        24.05.2025
# Licence:     MIT
# Version:     v1.0
# Description: train a neural network with AlexNet layer structure for AlexNet challenge to recognise images

# =====================================================================================================================================
# === Libraries
# =====================================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import gc
import time
import logging
# import json
# from PIL import Image as image

# =====================================================================================================================================
# === log GPU usage
# =====================================================================================================================================
# ToDo

# =====================================================================================================================================
# === Configure logfile
# =====================================================================================================================================
logfile = logging.getLogger(__name__)
logging.basicConfig(filename='output.log', level=logging.INFO)

# =====================================================================================================================================
# === Configure GPU (when available): Allocate 11GB
# =====================================================================================================================================
if tf.config.list_physical_devices('GPU'):
    tf.config.set_logical_device_configuration(
        tf.config.list_physical_devices('GPU')[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024*11)]
    )

logical_gpus = tf.config.list_logical_devices('GPU')
logfile.info( str( len(tf.config.list_physical_devices('GPU')) ) + "Physical GPU," +  str(len(logical_gpus)) + "Logical GPUs")

# =====================================================================================================================================
# === Constants
# =====================================================================================================================================
AMOUNT_FILES     = 289             # 289   # amount of train npy files (5 spltitted up to 289)
AMOUNT_VAL_FILES = 39              #  39   # amount of val   npy files (1 spltitted up to 39)
EPOCHS           = 30              # =50   # epochs in total

PATH             = "/mnt/scratch/data/imagenet-1k/"
DIRPATH          = "/home/alex/alexnet/"

# =====================================================================================================================================
# === Define model structure
# =====================================================================================================================================
alexNet = keras.models.Sequential([

    # Layer 0
    #keras.layers.Resizing(224, 224, interpolation="bilinear", input_shape=img.shape[1:] ) # resize to 227x227

    # Layer 1
    keras.layers.Conv2D(96, (11, 11), activation='relu', strides = 4, padding = "same", input_shape=(227,227,3)),  # A convolutional layer with 32 filters, a (3,3) kernel and relu activation
    keras.layers.MaxPooling2D(pool_size=(3,3), strides = 2),                    # A max-pooling layer with (2,2) binning

    # Layer 2
    keras.layers.Conv2D(256, (5, 5), activation='relu', padding = "same"),      # A convolutional layer with 256 filters, a (3,3) kernel and relu activation
    keras.layers.MaxPooling2D(pool_size=(3,3), strides = 2),                    # A max-pooling layer with (2,2) binning

    # Layer 3
    keras.layers.Conv2D(384, (3, 3), activation='relu', padding = "same"),      # A convolutional layer with 64 filters, a (3,3) kernel and relu activation
    keras.layers.Conv2D(384, (3, 3), activation='relu', padding = "same"),      # A convolutional layer with 64 filters, a (3,3) kernel and relu activation
    keras.layers.Conv2D(256, (3, 3), activation='relu', padding = "same"),      # A convolutional layer with 64 filters, a (3,3) kernel and relu activation
    keras.layers.MaxPooling2D(pool_size=(3,3), strides = 2),                    # A max-pooling layer with (2,2) binning

    # Layer 4
    keras.layers.Flatten(),                                                     # A flattening layer

    # Layer 5
    keras.layers.Dense(4096, activation='relu'),                                # A dense layer with 4096 nodes and relu activation
    keras.layers.Dropout(0.5),                                                  # A dropout layer with drop rate = 0.5 

    # Layer 6
    keras.layers.Dense(4096, activation='relu'),                                # A dense layer with 4096 nodes and relu activation
    keras.layers.Dropout(0.5),                                                  # A dropout layer with drop rate = 0.5

    # Layer 7
    keras.layers.Dense(1000, activation='softmax')                              # A softmaxed output layer
])

alexNet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
logfile.info("model is compiled")

# =====================================================================================================================================
# === Functions
# =====================================================================================================================================
def createGIF():
    return

def trainModel(model, num):

    # if model != MODEL_TYPE:
    #   raise ValueError("model not correct object type")

    if num not in range(0, AMOUNT_FILES):
        raise ValueError("File not found")

    # load data
    trainImages = np.load(PATH + "train/temp"   + str(num) +".npy", mmap_mode="r")
    trainLabels = np.load(PATH + "labels/labels"+ str(num) +".npy", mmap_mode="r")

    trainBatchImages = np.split(trainImages, 2)
    trainBatchLabels = np.split(trainLabels, 2)

    # train model
    model.train_on_batch(trainBatchImages[0], trainBatchLabels[0])
    model.train_on_batch(trainBatchImages[1], trainBatchLabels[1])

    # delete temorary data
    del trainLabels, trainImages, trainBatchImages, trainBatchLabels
    gc.collect()

    return

def calcPerformance():

        loss_list,     accu_list = [], []
    val_loss_list, val_accu_list = [], []

    # training accuracy, store results


    # validate accuracy, store results
    # Load test images and labels
    for num in range(0, AMOUNT_VAL_FILES):

        testImages = np.load(DIRPATH + "val/val_image"+ str(num)+".npy", mmap_mode="r")
        testLabels = np.load(DIRPATH + "val/val_labels"+str(num)+".npy", mmap_mode="r")

        results = alexNet.test_on_batch(testImages, testLabels)
        val_loss_list.append(results[0])
        val_accu_list.append(results[1])

        # deallocate data
        del testImages, testLabels
        gc.collect()

    # process result of batches
    results = [np.min(val_loss_list), np.max(val_loss_list), np.median(val_loss_list), np.min(val_accu_list), np.max(val_accu_list), np.median(val_accu_list)]
    return results

def saveImage(mylist, epoch):

    plt.figure(figsize=(14, 5))

    ## Plotting the training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(mylist['Accuracy (Min)'], 'bo-', label='Accuracy (Min)')  # Plot validation accuracy
    plt.plot(mylist['Accuracy (Max)'], 'ro-', label='Accuracy (Max)')  # Plot validation accuracy
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting the training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(mylist['Loss (Min)'], 'bo-', label='Loss (Min)')  # Plot validation accuracy
    plt.plot(mylist['Loss (Max)'], 'ro-', label='Loss (Max)')  # Plot validation accuracy
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.savefig(DIRPATH + "plots/plot"+ epoch +".jpg")
    plt.close()

    return

# =====================================================================================================================================
# === MAIN: Training
# =====================================================================================================================================
columns      = ['Loss (Min)', 'Loss (Max)', 'Loss (Median)', 'Accuracy (Min)', 'Accuracy (Max)', 'Accuracy (Median)']
results_list = []

# Measure time
start = time.time()
logfile.info("Start time: "+ str(start))

# unload and load files N times
for run in range(0, EPOCHS):
    logfile.info("RUN: "+ str(run) )

    # load each file sperately into GPU
    for num in range(0, AMOUNT_FILES):
        logfile.info("NUM: "+ str(num))

        # Train model
        trainModel(alexNet, num)

    # Valitation
    results_list.append( calcPerformance() )
    logfile.info( str( results_list ) )

    # Save Data after each "epoch"
    logfile.info("Checkpoint "+ str(run) + " :" + str(round(time.time() - start)) )

    # Save data: draw graph, save to csv file
    df = pd.DataFrame(results_list, columns=columns)
    df.to_csv(DIRPATH + '/alexnet_history.csv', mode="a", index=False, header=False)
    saveImage(df, run)

# When finished: Save model and history
alexNet.save(DIRPATH + 'model/alexnet2.h5')

# Measure time
logfile.info("Time to run: "+ str(time.time() - start) )


