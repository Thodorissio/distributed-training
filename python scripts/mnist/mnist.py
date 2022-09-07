
import tensorflow as tf
import os
import json
from time import perf_counter
from tensorflow import keras
import mnist_setup

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


x_train, y_train , x_test ,y_test = mnist_setup.mnist_dataset()



with strategy.scope():
  model =mnist_setup.build_and_compile_cnn_model()
  

tic = perf_counter()

history = model.fit(x_train, y_train, epochs=15, batch_size =256,  verbose =1)


training_time = perf_counter() - tic
print(f'Total training time: {training_time} secs')
