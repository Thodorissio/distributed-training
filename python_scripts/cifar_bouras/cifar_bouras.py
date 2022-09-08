import tensorflow as tf
import os
import json
from time import perf_counter
from tensorflow import keras
import cifar_bouras_setup

"""
Remember to set the TF_CONFIG envrionment variable.

For example:

export TF_CONFIG='{"cluster": {"worker": ["10.1.10.58:12345", "10.1.10.250:12345"]}, "task": {"index": 0, "type": "worker"}}'
"""

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


NUM_GPUS = 2
BS_PER_GPU = 128
NUM_EPOCHS = 60

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]


batch_size = 32
worker_dataset ,x_train, y_train , x_test ,y_test = cifar_bouras_setup.cifar_dataset(batch_size)
K = len(set(y_train))
M=x_train[0].shape



with strategy.scope():
  model =cifar_bouras_setup.build_and_compile_cnn_model(K,M)
  
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
  width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
 
train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size
tic = perf_counter()

r = model.fit(train_generator, validation_data=(x_test, y_test),
              steps_per_epoch=steps_per_epoch, epochs=10)


training_time = perf_counter() - tic
print(f'Total training time: {training_time} secs')