
import tensorflow as tf
from numpy import mean
from numpy import std
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
 


# load train and test dataset
def fashion_mnist_dataset(batch_size):
  # load dataset
  (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
  # reshape dataset to have a single channel
  trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
  testX = testX.reshape((testX.shape[0], 28, 28, 1))
  # one hot encode target values
  trainY = to_categorical(trainY)
  testY = to_categorical(testY)

  # convert from integers to floats
  train_norm = trainX.astype('float32')
  test_norm = testX.astype('float32')
  # normalize to range 0-1
  train_norm = train_norm / 255.0
  test_norm = test_norm / 255.0
  # return normalized images

  train_dataset = tf.data.Dataset.from_tensor_slices(
      (train_norm, trainY)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset ,train_norm, trainY ,test_norm ,testY
	


def build_and_compile_cnn_model():
  model = Sequential()
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(10, activation='softmax'))
  # compile model
  opt = SGD(lr=0.01, momentum=0.9)
  model.summary()
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model