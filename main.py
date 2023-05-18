
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from model import selectModel
from sklearn.metrics import accuracy_score, classification_report



np.random.seed(1)

filepath = "model-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, save_weights_only=False, save_freq='epoch')


def getData(dir,classes=None,datapercent=1):
    
    img_size = (256, 256)
    batch_size = 32

    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=1-datapercent)
    
    
    generator = datagen.flow_from_directory(
        dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        classes = classes,
        subset='training'
    )
    
    return generator
    
def generateData(classes=None,randomNum=0):
    if randomNum != 0:
        subdirs = [d for d in os.listdir("train") if os.path.isdir(os.path.join("train", d))]
        classes = list(np.random.choice(subdirs, randomNum, replace=False))

    print(classes)

    train = getData('train',classes=classes)
    test = getData('test',classes=classes)
    return train,test
    
    
def trainModel(trainGenerator,testGenerator):
    numOutputs = testGenerator.num_classes

    model = selectModel('resNetModel',d1=256,d2=256,channels=3,outputs=numOutputs,blocks=5)


    model.compile(optimizer="adam",
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    print(len(model.layers()))
    history = model.fit(trainGenerator, validation_data=testGenerator,epochs=30, batch_size=32,callbacks=[checkpoint],verbose=1)

    model.summary()
    
trainGenerator,testGenerator = generateData(classes=None,randomNum=0)

trainModel(trainGenerator,testGenerator)

