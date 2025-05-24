# =====================================================================================================================================
# === About
# =====================================================================================================================================
# Name:        preprocess.py 
# Author:      AlexVeso
# Date:        24.05.2025
# Licence:     MIT
# Version:     v1.0
# Description: split the 5 files into 289 smaller batches so that it can be loaded into limited memory and to train a neural network  

# =====================================================================================================================================
# === Libraries
# =====================================================================================================================================
import gc                 #
import numpy as np        #
import matplotlib.pyplot as plt
import time
import psutil
import json
import os

# %%
# =====================================================================================================================================
# === Constants
# =====================================================================================================================================
#PATH             = "/mnt/share/nnds/imagenet-1k/"
PATH             = "/mnt/scratch/data/imagenet-1k/"
AMOUNT_FILES_NPY = 5

# %%
# =====================================================================================================================================
# === functions
# =====================================================================================================================================
def deallocVar( deallocVar ):
    del deallocVar
    gc.collect()
    #time.sleep(10)

# %%
# =====================================================================================================================================
# === Find out in how much files the 5 files can be splitted up
# =====================================================================================================================================

# get length of all five npy files
lengths = []
for i in range(0, 5):
    lengths.append( len(np.load(PATH + "train_images/train_images_"+str(i)+".npy", mmap_mode="r")) ) 

# search for chunk size that is divisible by amount of array length
# Minimum:    5 => 200GB /    5 ~= 40,0 GB
# Maximum: 2000 => 200GB / 2000 ~=  0,1 GB 
for i in range(5, 2000):
    if (sum(lengths) / i == int(sum(lengths) / i)):
        print(str(i) + " | " + str(sum(lengths) / i) + " |  File Size: ", round(200/i, 2), " GB")


# %%
# Choose 289 as AMOUNT_FILES:
AMOUNT_FILES = 289
CHUNK_SIZE   = int( sum(lengths) / AMOUNT_FILES ) # = 2182*2
print("")
print("Chunk size:        ", CHUNK_SIZE)
print("Arrays:            ", sum(lengths))
print("amount of files:   ", sum(lengths) / CHUNK_SIZE)
print("lengths:           ", lengths)

# %%
# =====================================================================================================================================
# === Training Images | Open file, split file, save to multiple files
# =====================================================================================================================================
count  = 0             # number of chunk for saving to file
idx    = 0             # 

for i in range(0, AMOUNT_FILES_NPY):

    print("-------------------------------------")
    print("--- RUN " + str(i)                    )
    print("-------------------------------------")

    # -------------------------------------------------------------------------------------------------------------------------------------
    # --- load file
    # -------------------------------------------------------------------------------------------------------------------------------------
    cacheFile = np.load(PATH + "train_images/train_images_"+str(i)+".npy", mmap_mode="r")

    # -------------------------------------------------------------------------------------------------------------------------------------
    # --- calc 
    # -------------------------------------------------------------------------------------------------------------------------------------
    chunks    = int(len(cacheFile) / CHUNK_SIZE)                 # number of pieces to split arrays
    remainder = len(cacheFile) - chunks * CHUNK_SIZE             # amount of array elements left at the end after splitting all arrays

    print("Arrays in file:         ", len(cacheFile[idx:]))                  
    print("amount of chunks:       ", chunks)                          # number of pieces to split arrays
    print("size of chunks:         ", chunks * CHUNK_SIZE)             # amount of arrays to spliting in array from file
    print("remainder:              ", remainder)                       # amount of array elements left at the end after splitting all arrays

    # -------------------------------------------------------------------------------------------------------------------------------------
    # --- merge remainder cache with first part of cacheFile and store this FIRST!
    # -------------------------------------------------------------------------------------------------------------------------------------
    if count > 0:                                                 # do not run this in the first loop!

        idx = CHUNK_SIZE - len(cacheRemainder)                              # calculate the first part of cacheFile that is part of chunk of remainder
        print("idx:                    ", idx)              
        if idx > CHUNK_SIZE:
            raise ValueError("idx not calculated correctly!")

        first_batch_len = int(len(cacheFile[:idx])) + int(len(cacheRemainder))  # merge remainder nparray with first part of cacheFile
        print("len of cacheRemainder:  ", len(cacheRemainder))    
        print("len of cacheFile[:idx]: ", len(cacheFile[:idx]))          
        print("len of first_batch:     ", first_batch_len)              
        if not first_batch_len == CHUNK_SIZE:
            raise ValueError("first_batch not calculated correctly!")

        writeNPdata = np.vstack((cacheRemainder, cacheFile[:idx]))
        print("Cache I:  ", cacheRemainder.shape)
        print("Cache II: ", cacheFile[:idx].shape)
        print("ToFile:   ", writeNPdata.shape)

        #np.save("/mnt/scratch/data/imagenet-1k/train/temp"+str(count)+".npy", writeNPdata)        
        count += 1

    # -------------------------------------------------------------------------------------------------------------------------------------
    # --- cacheRemainder
    # -------------------------------------------------------------------------------------------------------------------------------------
    cacheRemainder = cacheFile[(chunks*CHUNK_SIZE+idx):]         #
    print("cacheRemainder:         ", len(cacheRemainder))          
    if not len(cacheRemainder) == (remainder-idx):
        raise ValueError("remainder of chunks not calculated correctly!")

    # -------------------------------------------------------------------------------------------------------------------------------------
    # --- split array into N chuncks of size CHUNK_SIZE and 1 chunk of size REST_SIZE
    # -------------------------------------------------------------------------------------------------------------------------------------
    cacheToFile = cacheFile[idx:(chunks*CHUNK_SIZE+idx)]         # 
    print("cacheToFile:            ", len(cacheToFile))             
    if not len(cacheToFile) == chunks * CHUNK_SIZE:           
        raise ValueError("chunks not calculated correctly: ", len(cacheToFile), " != ", chunks * CHUNK_SIZE)   

    # -------------------------------------------------------------------------------------------------------------------------------------
    # --- split, store remainder into cache
    # -------------------------------------------------------------------------------------------------------------------------------------
    batches = np.array_split(cacheToFile, chunks)
    print("amount of batches:      ", len(batches))
    if len(batches) not in range(chunks, chunks+2+1):
        raise ValueError("amount of batches not calculated correctly!")

    # -------------------------------------------------------------------------------------------------------------------------------------
    # --- store each batch onto file
    # -------------------------------------------------------------------------------------------------------------------------------------
    for batch in batches:

        if not len(batch) == CHUNK_SIZE:
            raise ValueError("Chunk Size is not "+ str(CHUNK_SIZE) +", but "+ str(len(batch)))

        #np.save("/mnt/scratch/data/imagenet-1k/train/temp"+str(count)+".npy", batch)
        count += 1

    print("COUNT:                  ", count)

    # -----------------------------------------------------------------------------
    # --- free GPU and RAM
    # -----------------------------------------------------------------------------
    del cacheToFile, cacheFile, batches
    gc.collect()

# %%
# -----------------------------------------------------------------------------
# --- Split files again
# -----------------------------------------------------------------------------
split = 2
files = os.listdir(PATH + "train")

if not len(files) * split == int(len(files) * split):
    raise ValueError("Amount of files not divisible by ", str(split)," - Amount of files: ", str(len(files)) )

if not CHUNK_SIZE / split == int(CHUNK_SIZE / split):
    raise ValueError("CHUNK_SIZE not divisible by ", str(split)," - CHUNK_SIZE: ", str(CHUNK_SIZE) )

count = 0
for file in files:
    #split_arr = np.split(np.load("/mnt/scratch/data/imagenet-1k/train/"+str(file)), split)
    for i in range(0, split):
        print(count + i)
        #np.save("/mnt/scratch/data/imagenet-1k/train/"+str(count)+".npy", np.array(split_arr[i]) )
        #np.save("/mnt/scratch/data/imagenet-1k/train/"+str(count+i)+".npy", np.array(split_arr[i]) )
    count += 2

# %%
# =====================================================================================================================================
# === Label Train Images | Open file, split file, save to multiple files
# =====================================================================================================================================
# import os

# load labels
with open(PATH + "classes.json") as file:
    classes = json.load(file)                                

# dictionary: get position in list (int in range € [0,999]) from class name (string)
dictLabels = {}
for i, label in enumerate(classes.keys()):
    dictLabels[label] = i 

# open all label files, load them into one array    
labels, labelsLength = [], []
for i in range(0, 4+1):
    with open(PATH + "train_labels/labels_train_images_"+ str(i) +".txt") as file: 
        labelsArray = [line.strip() for line in file]
        labels.append( labelsArray )
        labelsLength.append( len(labelsArray) )

# all labes
trainLabels = np.concatenate(labels).flat                      # 2d array to 1d array

allLabels = []
for trainLabel in trainLabels:                                 # a bit slow, but it works
    allLabels.append([dictLabels[trainLabel]])

print(trainLabels[1:10])
print(allLabels[1:10])
print(os.system("head "+PATH +"train_labels/labels_train_images_0.txt"))

# split files, check that all batches have the same size
batches = np.array_split(allLabels, AMOUNT_FILES*1)            # split into 289*2 files
print(np.array(batches).shape)                                 # 578 files, 2182 arrays with training vector of length 1000
batches = np.squeeze(batches, axis=2)                          # remove unneccesary dimension 
print(batches.shape)                                           # (578, 2182, 1, 1000) => (578, 2182, 1000)

# Debugging: check that all batches have the same size
for batch in batches:
    if not len(batch) == CHUNK_SIZE/1:
        raise ValueError("Chunk Size is not "+ str(CHUNK_SIZE/2) +", but "+ str(len(batch)))

# Save all batches to file
#for count, batch in enumerate(batches):
    #np.save("/mnt/scratch/data/imagenet-1k/labels/labels"+str(count)+".npy", batch)

# %%
from PIL import Image as image

with open(PATH + "classes.json") as file:
    classes = json.load(file)                                

# dictionary: get position in list (int in range € [0,999]) from class name (string)
dictLabels = {}
for i, label in enumerate(classes.keys()):
    dictLabels[i] = label 

num = 0
trainImages = np.load(PATH + "train_images/train_images_" + str(num) +".npy", mmap_mode="r")    # time to load npy  file into GPU: ~ 2 min
trainLabels = np.load(PATH + "labels/labels" + str(num) +".npy", mmap_mode="r")    # time to load npy  file into GPU: ~ 2 min

trainImages = np.load(PATH + "val/train_images_" + str(num) +".npy", mmap_mode="r")    # time to load npy  file into GPU: ~ 2 min
trainLabels = np.load(PATH + "val/labels" + str(num) +".npy", mmap_mode="r")    # time to load npy  file into GPU: ~ 2 min


labelsArray = []
with open(PATH + "train_labels/labels_train_images_0.txt") as file: 
    labelsArray = [line.strip() for line in file]

for i in range(10):

    fig = plt.figure()
    plt.imshow(trainImages[i], cmap='gray')
    plt.show()

    print(classes[dictLabels[trainLabels[i]]])       # training labels from np array
    print(classes[labelsArray[i]])                   # original labels from txt file


# %%
# ===================================================================================
# === Validation labels
# ===================================================================================
PATH = "/mnt/scratch/data/imagenet-1k/"
import json
import numpy as np

# load labels
with open(PATH + "classes.json") as file:
    classes = json.load(file)                               

# dictionary: get position in list (int in range € [0,999]) from class name (string)
dictLabels = {}
for i, label in enumerate(classes.keys()):
    dictLabels[label] = i 

# open all label files, load them into one array    
val_labels = []
with open(PATH + "val/labels_val_images.txt") as file:         # time to load txt  file:          < 1 sec
    valLabelsArray = [[dictLabels[line.strip()]] for line in file]

# split files, check that all batches have the same size
batches = np.array_split(valLabelsArray, 39)                        # split into 39 files
batches = np.array(batches).T                                  # transform to horizontal, not vertical
batches = np.squeeze(batches, axis=0)                          # remove unneccesary dimension
batches = np.stack(batches, axis=1)                            # align array correctly: (39, 1259)
print(batches.shape)

# Save all batches to file
for count, batch in enumerate(batches):
    #np.save("/mnt/scratch/data/imagenet-1k/val/val_labels"+str(count)+".npy", batch)
    print("COUNT val_label: ", count)

# ===================================================================================
# === Validation images
# ===================================================================================
val_images = np.load(PATH + "val/val_images.npy", mmap_mode="r")
print(val_images.shape)

batches = np.array( np.array_split(val_images, 39) )               # split into 39 files
print(batches.shape)
                                                              
for count, batch in enumerate(batches):
    #np.save("/mnt/scratch/data/imagenet-1k/val/val_image"+str(count)+".npy", batch)
    print("COUNT val_image: ", count)

# %%
# ===================================================================================
# === Check lengths of labels and image files
# ===================================================================================
print("LABELS: ", labelsLength, " | ", sum(labelsLength))
print("IMAGES: ", lengths,      " | ", sum(lengths))

# %%
# ===================================================================================
# === Check lengths of validation labels and validation image files
# ===================================================================================
with open(PATH + "labels_val_images.txt") as file:              # time to load txt  file:          < 1 sec
    print( len([line.strip() for line in file]) ) 

print( len(np.load(PATH + "val_images.npy", mmap_mode="r")) ) 

# 25.06: 49101 - 49101        = 0 => OK

# ===================================================================================
# === Check lengths of test image files
# ===================================================================================
print( len(np.load(PATH + "test_images.npy", mmap_mode="r")) ) 



