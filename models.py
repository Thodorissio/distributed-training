from time import perf_counter
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import random

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

import nltk
from nltk.corpus import movie_reviews

from transformers import BertTokenizer, TFBertForSequenceClassification

from typing import Tuple

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
AUTOTUNE = tf.data.AUTOTUNE

#We set the seed value so as to have reproducible results
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

class Cifar_10():

    def __init__(self, batch_size: int, epochs: int) -> None:
        
        self.batch_size = batch_size
        self.epochs = epochs


    def dataset(self) -> Tuple[tf.data.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """loads the cifar_10 dataset

        Returns:
            Tuple[tf.data.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                whole training dataset as well as x,y train and test sets
        """
        
        cifar10 = tf.keras.datasets.cifar10
        
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train, x_test = x_train / 255.0, x_test / 255.0

        y_train, y_test = y_train.flatten(), y_test.flatten()
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(self.batch_size)

        return train_dataset, x_train, y_train, x_test, y_test

    def model(self, inp_shape: Tuple[int, int, int, int], out_shape: int) -> tf.keras.Model:
        """Builds cifar model

        Args:
            inp_shape (Tuple[int, int, int, int]): shape of input instance
            out_shape (int): shape of output instance

        Returns:
            tf.keras.Model: compiled model
        """
        
        inp = Input(shape=inp_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Flatten()(x)
        x = Dropout(0.2)(x)

        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(out_shape, activation='softmax')(x)
        
        model = Model(inp, x)

        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        return model
    
    def fit_model(self) -> Tuple[float, float, int, int]:
        """Trains the cifar_10 model

        Returns:
            Tuple[float, float, int, int]: total training time, final training accuracy, trainable params and total params
        """

        train_dataset, x_train, y_train, _, _ = self.dataset()
        train_dataset = train_dataset.with_options(options)
        img_shape = x_train[0].shape
        classes = len(np.unique(y_train))

        model = self.model(inp_shape=img_shape, out_shape=classes)
        trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        tic = perf_counter()
        history = model.fit(train_dataset, batch_size=self.batch_size, epochs=self.epochs, steps_per_epoch=x_train.shape[0] // self.batch_size)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy, trainable_params, total_params


class Bert_movies():

    def __init__(self, batch_size: int, epochs: int) -> None:
        
        self.batch_size = batch_size
        self.epochs = epochs
        nltk.download('movie_reviews')

    def dataset(self) -> tf.data.Dataset:
        """loads and preprocess the nltk movie reviews dataset

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset: training dataset and testing tensor datasets
        """

        documents = [(' '.join(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

        train_set, _ = train_test_split(documents, test_size=0.1, random_state=42)

        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        X_train_text = [text for (text, label) in train_set]
        y_train = [1 if label == 'pos' else 0 for text, label in train_set]

        y_train = np.array(y_train)

        def convert_example_to_feature(review):
            return bert_tokenizer.encode_plus(review, 
                            add_special_tokens = True,    
                            max_length = 512,              
                            padding='max_length',
                            truncation=True,
                            return_attention_mask = True,  
                        )
        
        def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_masks,
            }, label

        def encode_examples(ds):
            # prepare list, so that we can build up final TensorFlow dataset from slices.
            input_ids_list = []
            token_type_ids_list = []
            attention_mask_list = []
            label_list = []
            for review, label in ds:
                bert_input = convert_example_to_feature(review)
                input_ids_list.append(bert_input['input_ids'])
                token_type_ids_list.append(bert_input['token_type_ids'])
                attention_mask_list.append(bert_input['attention_mask'])
                label_list.append([label])

            return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

        ds_train = zip(X_train_text, y_train)
        ds_train_encoded = encode_examples(ds_train).shuffle(len(X_train_text)).batch(self.batch_size)

        return ds_train_encoded

    def model(self) -> tf.keras.Model:
        """builds bert model and classifier

        Returns:
            tf.keras.Model: built model
        """

        bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        for i,encoder_layer in enumerate(bert_model.layers[0].encoder.layer):
            if i < len(bert_model.layers[0].encoder.layer) - 1:
                encoder_layer.trainable = False
            else:
                encoder_layer.trainable = True
        bert_model.layers[0].embeddings.trainable = False
        bert_model.layers[0].pooler.trainable = False
        bert_model.summary()
        
        return bert_model

    def fit_model(self) -> Tuple[float, float, int, int]:
        """Trains the bert model on movie reviews data

        Returns:
            Tuple[float, float, int, int]: total training time, final training accuracy, trainable params and total params
        """
        train_data = self.dataset()
        model = self.model()

        trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        learning_rate = 2e-4

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(loss=loss,  optimizer=optimizer,  metrics=metric)

        tic = perf_counter()
        history = model.fit(train_data, epochs=self.epochs, batch_size=self.batch_size)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy, trainable_params, total_params


class Natural_images_densenet():

    def __init__(self, batch_size: int, epochs: int) -> None:
        
        self.batch_size = batch_size
        self.epochs = epochs
    
    def dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """creates the data sets from the natural images dataset

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: data sets
        """

        data=[]
        labels=[]
        
        imagePaths = sorted(list(os.listdir("/home/user/distributed-training/datasets/natural_images/")))

        for img in imagePaths:

            path=sorted(list(os.listdir("/home/user/distributed-training/datasets/natural_images/"+img)))

            for i in path:
                image = cv2.imread("/home/user/distributed-training/datasets/natural_images/"+img+'/'+i)
                image = cv2.resize(image, (128,128))
                image = img_to_array(image)
                data.append(image)
                label = img
                labels.append(label)
        
        data = np.array(data, dtype="float32") / 255.0
        labels = np.array(labels)
        mlb = LabelBinarizer()
        labels = mlb.fit_transform(labels)

        (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)

        return x_train , x_test, y_train, y_test
    
    def model(self) -> tf.keras.Model:
        """creates the model which is based on DenseNet121

        Returns:
            tf.keras.Model: densenet model with 2 added layers
        """

        densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

        for layer in densenet.layers[:-2]:
            layer.trainable=False

        x = GlobalAveragePooling2D()(densenet.output)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x) 
        x = Dense(512, activation='relu')(x) 
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        preds = Dense(8, activation='softmax')(x)
        model = Model(inputs=densenet.input, outputs=preds)

        model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
        
        return model
    
    def fit_model(self) -> Tuple[float, float, int, int]:
        """Trains the densenet model on natural images data

        Returns:
            Tuple[float, float, int, int]: total training time, final training accuracy, trainable params and total params
        """
        
        x_train, _, y_train, _ = self.dataset()

        data_gen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip=True, shear_range=0.2)
        data_gen.fit(x_train)
        training_data = data_gen.flow(x_train, y_train, batch_size=self.batch_size)

        model = self.model()
        trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        tic = perf_counter()
        history = model.fit(training_data, batch_size=self.batch_size, epochs=self.epochs, steps_per_epoch=x_train.shape[0] // self.batch_size)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy, trainable_params, total_params
        

class Fashion_mnist():

    def __init__(self, batch_size: int, epochs: int) -> None:
        
        self.batch_size = batch_size
        self.epochs = epochs
    
    def dataset(self) -> Tuple[tf.data.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """loads fashion mnist data and creates dataset

        Returns:
            Tuple[tf.data.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                whole training dataset as well as x,y train and test sets
        """

        (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

        train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
        test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)

        train_norm = train_x.astype('float32')
        test_norm = test_x.astype('float32')

        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_norm, train_y)).shuffle(60000).repeat().batch(self.batch_size)

        return train_dataset, train_norm, train_y, test_norm, test_y
    
    def model(self) -> tf.keras.Model:
        """creates the model for the fashion mnist training

        Returns:
            tf.keras.Model: model
        """

        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))

        opt = SGD(lr=0.01, momentum=0.9)

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def fit_model(self) -> Tuple[float, float, int, int]:
        """Trains the fashion-mnist model that was created

        Returns:
            Tuple[float, float, int, int]: total training time, final training accuracy, trainable params and total params
        """
        
        train_dataset, x_train, _, _, _ = self.dataset()
        train_dataset = train_dataset.with_options(options)

        model = self.model()
        trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        tic = perf_counter()
        history = model.fit(train_dataset, batch_size=self.batch_size, epochs=self.epochs, steps_per_epoch=x_train.shape[0] // self.batch_size)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy, trainable_params, total_params


class Mnist_restnet():

    def __init__(self, batch_size: int, epochs: int) -> None:
        
        self.batch_size = batch_size
        self.epochs = epochs
    
    def dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """loads mnist data and creates dataset

        Returns:
            Tuple[tf.data.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                whole training dataset as well as x,y train and test sets
        """

        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

        x_train = np.expand_dims(x_train, axis=-1)
        x_train = np.repeat(x_train, 3, axis=-1)
        x_train = x_train.astype('float32') / 255
        x_train = tf.image.resize(x_train, [32,32])

        y_train = tf.keras.utils.to_categorical(y_train , num_classes=10)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(60000).repeat().batch(self.batch_size)

        return train_dataset, x_train
    
    def model(self) -> tf.keras.Model:
        """Loads resnet50 model and compiles it

        Returns:
            tf.keras.Model: model
        """

        inp = tf.keras.Input(shape=(32,32,3))

        resnet_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                    include_top = False, 
                                                    input_tensor = inp)

        x = tf.keras.layers.GlobalMaxPooling2D()(resnet_model.output)
        output = tf.keras.layers.Dense(10, activation='softmax', use_bias=True)(x)

        model = tf.keras.Model(resnet_model.input, output)

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def fit_model(self) -> Tuple[float, float, int, int]:
        """Trains the mnist model with resnet50

        Returns:
            Tuple[float, float, int, int]: total training time, final training accuracy, trainable params and total params
        """
        
        train_dataset, x_train = self.dataset()
        train_dataset = train_dataset.with_options(options)

        model = self.model()
        trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        tic = perf_counter()
        history = model.fit(train_dataset, batch_size=self.batch_size, epochs=self.epochs, steps_per_epoch=x_train.shape[0] // self.batch_size)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy, trainable_params, total_params