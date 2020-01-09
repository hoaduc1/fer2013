from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
import numpy as np

from utils import load_data, plot_dataset, plot_training_history

# read dataset and split it into input set and labels set
x_train, y_train = load_data();
print(x_train.shape)
print(y_train.shape)
''' fdsfdfffdf
ffdfdfdff
ffdfdff
'''''
# plot the dataset
#plot_dataset(x_train, y_train)

# pre-process
n_train=20000
x_train, x_test = x_train[:n_train, :], x_train[n_train:,:]
y_train, y_test = y_train[:n_train], y_train[n_train:]

y_train, y_test = to_categorical(y_train, 7), to_categorical(y_test, 7)

# reshape to be [samples][width][height][channels]
x_train = x_train.reshape((x_train.shape[0], 48, 48, 1)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], 48, 48, 1)).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

# create model
model = Sequential()
model.add(Conv2D(30, (3, 3), input_shape=(48, 48, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(15, (3, 3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))
model.add(Flatten(input_shape=(48, 48)))
model.add(Dense(128, activation='relu'))
# model.add(Dense(50, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(lr=0.001), loss=mean_squared_error, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=500, epochs=10, validation_data=(x_test, y_test))

plot_training_history(history)
	
