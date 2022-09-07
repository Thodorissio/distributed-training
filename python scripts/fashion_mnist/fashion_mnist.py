import tensorflow as tf
import os
import json
from time import perf_counter
from tensorflow import keras
import fashion_mnist_setup

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
single_worker_dataset ,x_train, y_train , x_test ,y_test = fashion_mnist_setup.fashion_mnist_dataset(batch_size)



with strategy.scope():
  model =fashion_mnist_setup.build_and_compile_cnn_model()
  
tic = perf_counter()
history = model.fit(single_worker_dataset, epochs=10, batch_size=32,steps_per_epoch=70, validation_data=(x_test, y_test))
training_time = perf_counter() - tic
print(f'Total training time: {training_time} secs')