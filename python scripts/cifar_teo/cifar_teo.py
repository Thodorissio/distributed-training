
import tensorflow as tf
import os
import json
from time import perf_counter
from tensorflow import keras
import cifar_teo_setup

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


x_train, y_train , x_test ,y_test ,x_val ,y_val = cifar_teo_setup.cifar_dataset()



with strategy.scope():
  model =cifar_teo_setup.build_and_compile_cnn_model()
  

tic = perf_counter()

history = model.fit(
    x_train, 
    y_train,
    epochs=10, 
    validation_data = (x_val, y_val), 
    batch_size=128
    )


training_time = perf_counter() - tic
print(f'Total training time: {training_time} secs')
