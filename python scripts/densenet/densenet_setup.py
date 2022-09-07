import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow 

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import random
import cv2
import math
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split



from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

def natural_dataset():
  data=[]
  labels=[]
  random.seed(42)
  imagePaths = sorted(list(os.listdir("/content/drive/MyDrive/natural_images/"))) #thelei allagi sto path
  random.shuffle(imagePaths)
  print(imagePaths)

  for img in imagePaths:
      path=sorted(list(os.listdir("/content/drive/My Drive/natural_images/"+img)))
      for i in path:
          image = cv2.imread("/content/drive/My Drive/natural_images/"+img+'/'+i)
          image = cv2.resize(image, (128,128))
          image = img_to_array(image)
          data.append(image)
          l = label = img
          labels.append(l)
  
  data = np.array(data, dtype="float32") / 255.0
  labels = np.array(labels)
  mlb = LabelBinarizer()
  labels = mlb.fit_transform(labels)
  print(labels[0])
  (xtrain,xtest,ytrain,ytest)=train_test_split(data,labels,test_size=0.4,random_state=42)
  print(xtrain.shape, xtest.shape ,ytrain.shape)
  return xtrain , xtest ,ytrain ,ytest
  

	


def build_and_compile_cnn_model():
  model_d=DenseNet121(weights='imagenet',include_top=False, input_shape=(128, 128, 3)) 

  x=model_d.output

  x= GlobalAveragePooling2D()(x)
  x= BatchNormalization()(x)
  x= Dropout(0.5)(x)
  x= Dense(1024,activation='relu')(x) 
  x= Dense(512,activation='relu')(x) 
  x= BatchNormalization()(x)
  x= Dropout(0.5)(x)

  preds=Dense(8,activation='softmax')(x) #FC-layer
  model=Model(inputs=model_d.input,outputs=preds) #isos thelei anti gia model_d na balo base model

  for layer in model.layers[:-8]:
    layer.trainable=False
    
  for layer in model.layers[-8:]:
      layer.trainable=True
  model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
  model.summary()
  
  return model