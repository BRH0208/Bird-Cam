#Imports
from PIL import Image as PIL
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
import cv2 as cv
from os import path
from os import listdir
from os.path import isfile, join
import random
from math import ceil


#Much of this code was taken from https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71

##ConfigVars
datasetName = "Dataset"
saveLocation = "SavedModels\\Checkpoints\\cp-{epoch:04d}.ckpt"
exportLocation = "SavedModels\\ExportedModels\\model_export"
saveLoad = True #Look for a recent save. If it exists: Load it before training
exportWhenComplete = False
randomSeed = 12345
batchSize = 128
imgSize = (32,32) ##32 v 32 reccomended
numEpochs = 50 #0 for no training(Usefull for exporting)
sharedVerbose = 1
isGrayScale = True 
splitPercentage = 0.05
internalLayers = [64,64]
dropoutAmount = 0.35 #Used to prevent overfitting. 0.25 is more than plenty, 0 to disable
camNum = 0

##Session Management
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#session = tf.compat.v1.Session(config=config)

##Debug Setup
print('Pillow Version:', PIL.__version__)
print("Tensorflow is running at: ",tf.__version__)
print("Eager execution: ",tf.executing_eagerly())
filePath = (path.dirname(path.realpath(__file__)))
print("Applicaiton running in "+filePath)
saveDir = path.dirname(saveLocation)
print("Saving to "+saveDir)
print("Loading From "+filePath+"\\"+saveLocation)
print("Exposed GPU's "+str(tf.config.experimental.list_physical_devices('GPU')))
print("Exposed CPU's "+str(tf.config.experimental.list_physical_devices('CPU')))
print()
print("Looking for dataset directory: ",end="")

##Get files
directories = [f for f in listdir(filePath) if path.isdir(join(filePath, f))]
assert len(directories) > 0
print("Found "+str(len(directories))+" directories")
assert datasetName in directories
for directory in directories:
    print(" - "+str(directory),end="")
    if(str(directory) == datasetName):
        print("*")        
    else:
        print()
print()
print("Getting Classifiers "+str(datasetName))
directories = [f for f in listdir(filePath+"\\"+datasetName) if path.isdir(join(filePath+"\\"+datasetName, f))]
for directory in directories:
    print(" - "+str(directory) + " > "+str(len([f for f in listdir(filePath+"\\"+datasetName+"\\"+str(directory)) if path.isfile(join(filePath+"\\"+datasetName+"\\"+str(directory), f))])) + "# files")
    
print()
##Actual Doing stuff
print("Turning Data into an array")
##Of course not a np array I hate those and won't use them unless I have to, even if they are more effecient. 
##If you say this is a list not an array I will trap you inside this program for a thousand years!

masterFileArray = []
for i in range(len(directories)):
    files = [f for f in listdir(filePath+"\\"+datasetName+"\\"+str(directories[i])) if path.isfile(join(filePath+"\\"+datasetName+"\\"+str(directories[i]), f))]
    if(len(directories) != 2):
        for file in files:
            oneHot = [0] * len(directories)
            oneHot[i] = 1
            masterFileArray.append((filePath+"\\"+datasetName+"\\"+str(directories[i])+"\\"+str(file),oneHot))
    else:
        for file in files:
            oneHot = i
            masterFileArray.append((filePath+"\\"+datasetName+"\\"+str(directories[i])+"\\"+str(file),oneHot))
print("Shuffling.")
random.seed(randomSeed) #Setting a seed allows re-training the same model without contaminiating the test dataset
random.shuffle(masterFileArray)

print(str(len(masterFileArray)) + "# Array Elements") 
for i in range(min(len(masterFileArray),10)):
    print(str(i) + " : "+ str(masterFileArray[i]))

        
testAmount = int(splitPercentage*len(masterFileArray))
print("Setting aside "+str(testAmount)+" as testing ("+str(splitPercentage)+" of total)")
testFiles = masterFileArray[:testAmount]
trainFiles = masterFileArray[testAmount:]
print("testFiles : "+str(len(testFiles)))
print("trainFiles : "+str(len(trainFiles)))
print()

##Functions & Classes

class KerasGeneratorFromFile(keras.utils.Sequence) :

    def __init__(self, fileArray, batch_size) :
        self.fileArray = fileArray
        self.batch_size = batch_size

    def __len__(self):
        return int(ceil(float(len(self.fileArray)) / float(self.batch_size)))

    def __getitem__(self,idx):
        #assert (((idx+1) * self.batch_size) <= len(self.fileArray))
        data = []
        labels = []
        for fileID in range((idx * self.batch_size),((idx+1) * self.batch_size)-1):
            if(fileID >= len(self.fileArray)):
                break
            data.append(prepareFileImage(self.fileArray[fileID][0]))
            labels.append(self.fileArray[fileID][1]) ##This is run every time a batch is requested, it is run a bunch and it isn't the fastest. Too Bad!
        if(len(directories) == 2):
            trueY = np.array(labels)
        else:
            trueY = np.array(labels)
        
        if(isGrayScale):
            trueX = np.reshape(np.array(data),(len(data),imgSize[0],imgSize[1],1))
        else:
            trueX = np.reshape(np.array(data),(len(data),imgSize[0],imgSize[1],3))
        return (trueX,trueY)
        
    
def prepareArrayImage(array):
    return prepareImage(PIL.fromarray(array))
def prepareImage(img):
    if(isGrayScale):
        return np.array(img.convert('L').resize(imgSize))
    else:
        return np.array(img.convert('RGB').resize(imgSize))

def prepareFileImage(imgPath):
    return prepareImage(PIL.open(imgPath))


#Creating Custom Generators
print("Creating Generators")
trainGen = KerasGeneratorFromFile(trainFiles,batchSize)
testGen = KerasGeneratorFromFile(testFiles,batchSize)
print("Testing Generator : ",end="")
sampleImages = trainGen.__getitem__(int(len(trainFiles) // batchSize))
print("("+str(sampleImages[0].shape)+","+str(sampleImages[1].shape)+") shape returned")
print(sampleImages[0][0])
print(sampleImages[1][0])



    
print("Generating Model")
model = Sequential()


#Convo2D

if(isGrayScale):
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation ='relu',input_shape=(imgSize[0],imgSize[1],1)))
else:
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation ='relu',input_shape=(imgSize[0],imgSize[1],3)))
model.add(BatchNormalization())
model.add(Dropout(dropoutAmount))
model.add(MaxPooling2D(pool_size=(2,2)))

#Conv2d #2
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropoutAmount))
model.add(MaxPooling2D(pool_size=(2,2)))

#Con2d #3
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropoutAmount))
model.add(Flatten())

#Start of regular Network

#Dense First
model.add(Dense(64, activation = "relu")) ##Input Layer
model.add(BatchNormalization())
model.add(Dropout(dropoutAmount))

for layer in internalLayers:#Hidden Layers
    model.add(Dense(layer, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropoutAmount))

#Output Layer
if(len(directories) != 2):
    model.add(Dense(len(directories), activation = "softmax")) #Classification layer or output layer
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])#Finish the model

else:
    model.add(Dense(1, activation = "sigmoid")) #Classification layer or output layer
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])#Finish the model




#Save initial
#model.save_weights(filePath+"\\"+saveLocation.format(epoch=0))

model.summary()

#Prepare for saving
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saveLocation,
                                                 save_weights_only=True,
                                                 verbose=1)


##Attempt to load data
if(saveLoad):
    print("Attempting to load previous weights")
    try:
        latest = tf.train.latest_checkpoint(saveDir)
        model.load_weights(latest)
        loss, acc = model.evaluate(testGen, verbose=sharedVerbose)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    except:
        print("Error in loading: Skipping Loading")
        saveLoad = False
if(not(saveLoad)):
    model.save_weights(saveLocation.format(epoch=0))

print()
if(numEpochs != 0):
    print("Training Start")
    #Training Time
    model.fit(x=trainGen,
        steps_per_epoch = int(len(trainFiles) // batchSize),
        epochs = numEpochs,
        verbose = sharedVerbose,
        validation_data = testGen,
        validation_steps = int(len(testFiles) // batchSize),
        callbacks=[cp_callback])
else:
    print("Training Skipped")
#Optional Model Export
if(exportWhenComplete):
    print("Exporting Model")
    model.save(exportLocation)

print("Starting Demo")
##This video camera demo tends to not work as your camera input isn't like the training data at all. It would be fun if it worked but it doesn't

capture = cv.VideoCapture(camNum)
while True:
    isTrue,frame = capture.read()
    picture = prepareArrayImage(np.asarray(frame))
    if(isGrayScale):
        picture = np.reshape(np.array(picture),(imgSize[0],imgSize[1],1))
    else:
        picture = np.reshape(np.array(picture),(imgSize[0],imgSize[1],3))
    cv.imshow('frame',cv.resize(np.uint8(np.array(picture)), (500, 500)))#1000 Output Size
    picture=np.array([picture])
    predictions = model.predict(picture)
    predictions = predictions.tolist()
    highestConfidence = max(predictions[0])#Predictions a 1 element array don't question it
    highestIndex = predictions[0].index(highestConfidence)
    if(highestConfidence < 0.5) :
        print("None (",end="")
        print(str(directories[highestIndex]) +" "+ str(int(1000*highestConfidence)/10)+"%)")
    else:
        print(str(directories[highestIndex]) +" "+ str(int(1000*highestConfidence)/10)+"%")
    
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
'''
##This alternative demo usesing test files. Ensure your trained model is on the same shuffle seed to prevent contamination
for batchNum in range(int(len(testFiles)/batchSize)):
    batchImg,batchLabel = testGen.__getitem__(batchNum)
    predictions = model.predict(batchImg)
    for i in range(len(batchImg)):
        highestConfidence = max(predictions[i])
        highestIndex = predictions[i].tolist().index(highestConfidence)
        predictedCatagory = str(directories[batchLabel[i].tolist().index(1)]) + " : "
        if(highestConfidence < 0.5) :
            predictedCatagory += "None ("+str(directories[highestIndex]) +" "+ str(int(10000*highestConfidence)/100)+"%)"
        else:
            predictedCatagory += str(directories[highestIndex]) +" "+ str(int(10000*highestConfidence)/100)+"%"
        cv.imshow(predictedCatagory,cv.resize(np.uint8(batchImg[i]), (600, 600)))
        print("Showing "+str(i)+" predicted as "+str(predictedCatagory))
        cv.waitKey(0)##Wait indefinitly until a key is pressed5
    print(" -- batchNum: "+str(batchNum))
print("All Done")

'''
