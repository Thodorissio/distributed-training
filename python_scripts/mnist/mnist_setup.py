import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from time import perf_counter
import numpy as np
import random
import matplotlib.pyplot as plt

def mnist_dataset():

  mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
  mnist.data.shape

  
  X = mnist.data.astype('float32')
  y = mnist.target.astype('int64')
  X /= 255.0
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
  return X_train , y_train ,X_test , y_test

def build_and_compile_cnn_model():
    inp = tf.keras.Input((784,))

    x = tf.keras.layers.Dense(16, activation='relu')(inp)
    x = tf.keras.layers.Dropout(0.05)(x)
    out = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(
        inputs=inp,
        outputs=out,
    )
  
  # model description
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                )
    return model
