# Import libraries
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import numpy as np

##Step 1: Remove line 13 and load X_test as arrays of image size(28*28) * number of images
##Step 2: If image size is different change num_pixels

#load X_test and reshape into 28*28 pixels
##Load only X_test and remove below line
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = 28*28

#flatten 28*28 images to a 784 vector for each image
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

# normalize inputs from 0-255 to 0-1
X_test = X_test / 255

# model
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(300, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

#load weights
model.load_weights("SNN_Weights.h5")

#predict a number
prediction = np.asarray(model.predict(X_test))
indices = np.argmax(prediction, axis=1)
numbers = indices + 1
print(numbers)
