from time import perf_counter
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

from typing import Tuple

class Cifar_10():

    def __init__(self, batch_size: int) -> None:
        
        self.batch_size = batch_size


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
        """Trains the cifar_10 model with multiworker mirrored strategy

        Returns:
            Tuple[float, float]: total training time and final training accuracy
        """

        dataset, x_train, y_train, x_test, y_test = self.dataset()
        img_shape = x_train[0].shape
        classes = len(np.unique(y_train))

        cifar_model = self.model(inp_shape=img_shape, out_shape=classes)
        
        tic = perf_counter()
        history = cifar_model.fit(dataset, epochs=5, steps_per_epoch=70)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy


class IMDB_sentiment():

    def init(self, batch_size: int) -> None:

        self.batch_size = batch_size

    def dataset(self):

        df = pd.read_csv("./IMDB Dataset.csv")

        tokenizer = BertTokenizer.from_pretrained("bert-case-uncased")

        df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

        train_df = df[:45000] 
        test_df = df[45000:]
        
        def convert_data_to_tf_data(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
            
            examples_df = df.apply(lambda x: InputExample(
                                                guid=None, 
                                                text_a = x['review'], 
                                                label = x['sentiment']), 
                                                axis = 1,
                                            )

            return examples_df

        

        