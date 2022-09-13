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

from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

from transformers import AutoTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

from typing import Tuple

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

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
        
        # Load in the data
        cifar10 = tf.keras.datasets.cifar10
        
        # Distribute it to train and test set
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Reduce pixel values
        x_train, x_test = x_train / 255.0, x_test / 255.0
        
        # flatten the label values
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
        
        # Hidden layer
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # last hidden layer i.e.. output layer
        x = Dense(out_shape, activation='softmax')(x)
        
        model = Model(inp, x)

        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        return model
    
    def run_model(self) -> Tuple[float, float]:
        """Trains the cifar_10 model

        Returns:
            Tuple[float, float]: total training time and final training accuracy
        """

        train_dataset, x_train, y_train, _, _ = self.dataset()
        train_dataset = train_dataset.with_options(options)
        img_shape = x_train[0].shape
        classes = len(np.unique(y_train))

        cifar_model = self.model(inp_shape=img_shape, out_shape=classes)
        
        tic = perf_counter()
        history = cifar_model.fit(train_dataset, batch_size=self.batch_size, epochs=self.epochs, steps_per_epoch=x_train.shape[0] // self.batch_size)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy


class IMDB_sentiment():

    def __init__(self, batch_size: int, epochs: int) -> None:
        
        self.batch_size = batch_size
        self.epochs = epochs

    def convert_data_to_tf_data(self, df: pd.DataFrame, tokenizer: AutoTokenizer) -> pd.DataFrame:
            
            examples_df = df.apply(lambda x: InputExample(
                                                guid=None, 
                                                text_a = x['review'], 
                                                label = x['sentiment']), 
                                                axis = 1,
                                            )
            
            features = [] 

            for e in examples_df:
                input_dict = tokenizer.encode_plus(
                    e.text_a,
                    add_special_tokens=True,    
                    max_length=128,    
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    pad_to_max_length=True, 
                    truncation=True
                )

                input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],input_dict["token_type_ids"], input_dict['attention_mask'])
                features.append(InputFeatures( input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label) )

            def gen():
                for f in features:
                    yield (
                        {
                            "input_ids": f.input_ids,
                            "attention_mask": f.attention_mask,
                            "token_type_ids": f.token_type_ids,
                        },
                        f.label,
                    )

            return tf.data.Dataset.from_generator(
                gen,
                ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
                (
                    {
                        "input_ids": tf.TensorShape([None]),
                        "attention_mask": tf.TensorShape([None]),
                        "token_type_ids": tf.TensorShape([None]),
                    },
                    tf.TensorShape([]),
                ),
            )

    def dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """loads the IMDB Dataset

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset: training dataset and testing tensor datasets
        """

        df = pd.read_csv("/home/user/distributed-training/datasets/IMDB Dataset.csv")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

        train_df = df[:45000] 
        test_df = df[45000:]

        train_data = self.convert_data_to_tf_data(df=train_df, tokenizer=tokenizer)
        train_data = train_data.batch(self.batch_size)

        test_data = self.convert_data_to_tf_data(df=test_df, tokenizer=tokenizer)
        test_data = test_data.batch(self.batch_size)

        return train_data, test_data
    
    def run_model(self) -> Tuple[float, float]:
        """Trains the bert transformer on IMBD_sentiment_analysis data

        Returns:
            Tuple[float, float]: total training time and final training accuracy
        """
        
        train_data, _ = self.dataset()
        train_data = train_data.with_options(options)
        
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

        #Only the last layer will be fine tuned cause of RAM limitations
        for layer in model.layers[:-1]:
            layer.trainable=False

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])


        tic = perf_counter()
        history = model.fit(train_data, batch_size=self.batch_size, epochs=self.epochs)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy


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

        x = GlobalAveragePooling2D()(densenet.output)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x) 
        x = Dense(512, activation='relu')(x) 
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        preds = Dense(8, activation='softmax')(x)
        model = Model(inputs=densenet.input, outputs=preds)

        for layer in model.layers[:-8]:
            layer.trainable=False
            
        for layer in model.layers[-8:]:
            layer.trainable=True

        model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
        
        return model
    
    def run_model(self) -> Tuple[float, float]:
        """Trains the densenet model on natural images data

        Returns:
            Tuple[float, float]: total training time and final training accuracy
        """
        
        x_train, _, y_train, _ = self.dataset()

        data_gen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip=True, shear_range=0.2)
        data_gen.fit(x_train)
        training_data = data_gen.flow(x_train, y_train, batch_size=self.batch_size)

        model = self.model()

        tic = perf_counter()
        history = model.fit(training_data, batch_size=self.batch_size, epochs=self.epochs, steps_per_epoch=x_train.shape[0] // self.batch_size)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy
        

class Fashion_mnist():

    def __init__(self, batch_size: int, epochs: int) -> None:
        
        self.batch_size = batch_size
        self.epochs = epochs
    
    def dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """loads fashion mnist data and creats dataset

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

    def run_model(self) -> Tuple[float, float]:
        """Trains the fashion-mnist model that was created

        Returns:
            Tuple[float, float]: total training time and final training accuracy
        """
        
        train_dataset, x_train, y_train, _, _ = self.dataset()
        train_dataset = train_dataset.with_options(options)

        model = self.model()

        tic = perf_counter()
        history = model.fit(train_dataset, batch_size=self.batch_size, epochs=self.epochs, steps_per_epoch=x_train.shape[0] // self.batch_size)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy