from __future__ import print_function
import numpy as np
import h5py

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

nb_classes = 10

model = Sequential()
input_shape = (1, img_rows, img_cols)

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid', activation='relu',
                        input_shape=input_shape))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.load_weights('MNIST_weights.h5')

# launching the visualization server
from quiver_engine import server
server.launch(model, temp_folder='./tmp', input_folder='./', port=7777)

