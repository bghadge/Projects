from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, MaxPooling2D
import numpy as np

#download mnist data and split into train and test sets
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#creating model
model = Sequential()

#add model layers
model.add(Conv2D(48, kernel_size=3, padding = 'valid', activation = 'relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(units=100,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

#compile the model
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.summary()

history = model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=10, batch_size=200, verbose=2)

#save weights
model.save("digit_classifier.h5")
