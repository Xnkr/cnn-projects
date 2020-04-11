# Imports
import tensorflow as     tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import math
import numpy             as np
import matplotlib.pyplot as plt

# ResNet32
n = 32
batch_size = 32
epochs = 50
num_classes = 10

# Checkpoint directory
SAVE_MODEL_PATH = './save/model/'

# Load dataset
dataset = cifar10.load_data()

x_train, y_train = dataset[0]
x_test, y_test = dataset[1]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

train_mean = x_train.mean()
train_std = x_train.std()

# Normalize data
x_train = (x_train - train_mean) / train_std
x_test = (x_test - train_mean) / train_std


def resnet(n):
    n_layers = (n - 2) // 6
    model_input = keras.Input(shape=(32,32,3), name='input_image')
    x = model_input
    
    x = keras.layers.Conv2D(16,3,strides=1,padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    for _ in range(n_layers):
        res = keras.layers.Conv2D(16,3,strides=1,padding='same')(x)
        res = keras.layers.BatchNormalization()(res)
        res = keras.layers.ReLU()(res)
        res = keras.layers.Conv2D(16,3,strides=1,padding='same')(res)
        res = keras.layers.BatchNormalization()(res)
        x = keras.layers.Add()([x, res])
        x = keras.layers.ReLU()(x)

    # Level 2

    for i in range(n_layers):
        strides = 2 if i == 0 else 1 # Down sampling
        res = keras.layers.Conv2D(32,3,strides=strides,padding='same')(x)
        res = keras.layers.BatchNormalization()(res)
        res = keras.layers.ReLU()(res)
        res = keras.layers.Conv2D(32,3,strides=1,padding='same')(res)
        res = keras.layers.BatchNormalization()(res)
        if i == 0:
            x = keras.layers.Conv2D(32,1,strides=strides)(x)
        x = keras.layers.Add()([x, res])
        x = keras.layers.ReLU()(x)

    
    # Level 3
    for i in range(n_layers):
        strides = 2 if i == 0 else 1 # Down sampling
        res = keras.layers.Conv2D(64,3,strides=strides,padding='same')(x)
        res = keras.layers.BatchNormalization()(res)
        res = keras.layers.ReLU()(res)
        res = keras.layers.Conv2D(64,3,strides=1,padding='same')(res)
        res = keras.layers.BatchNormalization()(res)
        if i == 0:
            x = keras.layers.Conv2D(64,1,strides=strides)(x)
        x = keras.layers.Add()([x, res])
        x = keras.layers.ReLU()(x)
    
    y = keras.layers.GlobalAveragePooling2D()(x)
    output = keras.layers.Dense(num_classes,activation='softmax')(y)

    model = keras.Model(inputs=model_input, outputs=output)
    return model

model = resnet(n)
# keras.utils.plot_model(model, 'resnet32.png', show_shapes=True)
model.summary()

# Learning rate schedule 
# Source: Lecture code
TRAINING_LR_MAX          = 0.001
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 5
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 55

TRAINING_NUM_EPOCHS = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT    = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL   = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

def lr_schedule(epoch):
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL


    return lr

callbacks = [keras.callbacks.LearningRateScheduler(lr_schedule),
             keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_PATH+'resnet_{epoch}.h5', monitor='val_loss', verbose=1)]

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])

# Data Augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
datagen.fit(x_train)

import pickle
history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              epochs=epochs,
              validation_data=(x_test, y_test))

pickle.dump(history.history, open('hist.p', 'wb'))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('acc.png')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss.png')

