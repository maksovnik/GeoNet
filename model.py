import tensorflow.keras as keras
from keras.layers import Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from keras.layers import MaxPooling2D, Flatten, Input, Reshape, Permute, multiply, Lambda
from keras.models import Model,Sequential

from keras import layers
from keras import models

def simpleModel(d1,d2,channels,outputs,numBlocks,attention):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(d1,d2,channels)),
        keras.layers.Dense(outputs, activation='softmax')
    ])
    
    return model


def convModel(d1, d2, channels, outputs,numBlocks,attention):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(d1, d2, channels)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(outputs, activation='softmax')
    ])

    return model


def selectModel(val,d1,d2,channels,outputs,blocks=0,attention=False):
    models = {'simpleModel':simpleModel,
              'convModel':convModel,
              'resNetModel':ResNet50,}
    return models[val](d1,d2,channels,outputs,blocks,attention)
    


def squeeze_excite_block(input_tensor, ratio=16):
    init = input_tensor
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = layers.multiply([init, se])
    return x

def identity_block(X, f, filters,attention=False):
    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)
    
    if attention:
        X = squeeze_excite_block(X)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, s=2,attention=False):
    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv2D(F1, (1, 1), strides=(s, s))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(F2, (f, f), strides=(1, 1), padding='same')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(F3, (1, 1), strides=(1, 1), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)

    if attention:
        X = squeeze_excite_block(X)

    X_shortcut = layers.Conv2D(F3, (1, 1), strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3)(X_shortcut)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def ResNet50(d1,d2,channels,outputs,numC,attention=False):
    input_tensor = layers.Input(shape=(d1,d2,channels))

    X = layers.ZeroPadding2D((3, 3))(input_tensor)

    # Stage 1
    X = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1,attention=attention)
    X = identity_block(X, 3, [64, 64, 256],attention=attention)
    X = identity_block(X, 3, [64, 64, 256],attention=attention)

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2,attention=attention)
    X = identity_block(X, 3, [128, 128, 512],attention=attention)
    X = identity_block(X, 3, [128, 128, 512],attention=attention)
    X = identity_block(X, 3, [128, 128, 512],attention=attention)

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2,attention=attention)
    X = identity_block(X, 3, [256, 256, 1024],attention=attention)
    X = identity_block(X, 3, [256, 256, 1024],attention=attention)
    X = identity_block(X, 3, [256, 256, 1024],attention=attention)
    X = identity_block(X, 3, [256, 256, 1024],attention=attention)
    X = identity_block(X, 3, [256, 256, 1024],attention=attention)

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2,attention=attention)
    X = identity_block(X, 3, [512, 512, 2048],attention=attention)
    X = identity_block(X, 3, [512, 512, 2048],attention=attention)

    X = layers.GlobalAveragePooling2D()(X)
    X = layers.Dense(outputs, activation='softmax')(X)

    model = models.Model(inputs=input_tensor, outputs=X, name='ResNet50')

    return model


model1 = selectModel('simpleModel',256,256,3,50,blocks=3,attention=False) # phase 1
model2 = selectModel('convModel',256,256,3,50,blocks=3,attention=False) # phase 2
model3 = selectModel('resNetModel',256,256,3,50,blocks=3,attention=False) # phase 3
model4 = selectModel('resNetModel',256,256,3,50,blocks=3,attention=True) # phase 4

models=[model1,model2,model3,model4]
# print(len(model.layers))

# dc = {}
# for layer in model.layers:
#     tp = layer.__class__.__name__
#     if tp in dc.keys():
#         dc[tp] +=1
#     else:
#         dc[tp] = 1

# print(dc)

# print(sum(dc.values()))

for model in models:
    print(model.count_params())

# print(len(model.trainable_weights))