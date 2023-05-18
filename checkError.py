import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import os
import numpy as np
from sklearn.metrics import classification_report

np.set_printoptions(suppress=True)

def getData(dir, classes=None, datapercent=1):
    img_size = (256, 256)
    batch_size = 32

    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=1 - datapercent)

    generator = datagen.flow_from_directory(
        dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=False,
        subset='training'
    )

    return generator

testGenerator = getData('test')

subdirs = [d for d in os.listdir("train") if os.path.isdir(os.path.join("train", d))]


def unOneHot(x):
    return subdirs[np.argmax(x)]


model = load_model('runOne/model-09.h5', compile=False)

model.compile(optimizer="adam",
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

predictions = model.predict(testGenerator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = testGenerator.classes

class_names = list(testGenerator.class_indices.keys())

accuracy = np.mean(predicted_labels == true_labels)

report = classification_report(true_labels, predicted_labels, target_names=subdirs,digits=4)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)