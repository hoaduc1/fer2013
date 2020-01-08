from keras.models import Sequential
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.regularizers import l2
import numpy as np

from utils import load_data, plot_dataset, plot_training_history

# read dataset and split it into input set and labels set
x_train, y_train = load_data()
print(x_train.shape)
print(y_train.shape)

# plot the dataset
#plot_dataset(x_train, y_train)

# pre-process
n_train=23000
x_train, x_test = x_train[:n_train, :], x_train[n_train:,:]
y_train, y_test = y_train[:n_train], y_train[n_train:]

y_train, y_test = to_categorical(y_train, 7), to_categorical(y_test, 7)

# reshape to be [samples][width][height][channels]
x_train = x_train.reshape((x_train.shape[0], 48, 48, 1)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], 48, 48, 1)).astype('float32')

# normalize inputs
x_train -= np.mean(x_train, axis=0)
x_train /= np.std(x_train, axis=0)

# create model
num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48, 48
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

#Compliling the model with adam optimixer and categorical crossentropy loss
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

#training the model
history = model.fit(np.array(x_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(x_test), np.array(y_test)),
          shuffle=True)

plot_training_history(history)
	
