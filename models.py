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

from transformers import AutoTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

from typing import Tuple

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

#We set the seed value so as to have reproducible results
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

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
        """Trains the cifar_10 model

        Returns:
            Tuple[float, float]: total training time and final training accuracy
        """

        train_dataset, x_train, y_train, x_test, y_test = self.dataset()
        train_dataset = train_dataset.with_options(options)
        img_shape = x_train[0].shape
        classes = len(np.unique(y_train))

        cifar_model = self.model(inp_shape=img_shape, out_shape=classes)
        
        tic = perf_counter()
        history = cifar_model.fit(train_dataset, epochs=5, steps_per_epoch=70)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy


class IMDB_sentiment():

    def __init__(self, batch_size: int) -> None:

        self.batch_size = batch_size

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
        """loads the cifar_10 dataset

        Returns:
            Tuple[tf.data.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                whole training dataset as well as x,y train and test sets
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
    
    def run_model(self):
        """Trains the bert transformer on IMBD_sentiment_analysis data

        Returns:
            Tuple[float, float]: total training time and final training accuracy
        """
        
        train_data, _ = self.dataset()

        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

        #Only the last layer will be fine tuned cause of RAM limitations
        for layer in model.layers[:-1]:
            layer.trainable=False

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])


        tic = perf_counter()
        history = model.fit(train_data, epochs=5)
        training_time = perf_counter() - tic
        
        training_accuracy = history.history['accuracy'][-1]

        return training_time, training_accuracy

        