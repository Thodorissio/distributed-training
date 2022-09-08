import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

def cifar_dataset(batch_size):
  # Load in the data
  cifar10 = tf.keras.datasets.cifar10
 
  # Distribute it to train and test set
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  # Reduce pixel values
  x_train, x_test = x_train / 255.0, x_test / 255.0
 
  # flatten the label values
  y_train, y_test = y_train.flatten(), y_test.flatten()
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)

 
  return train_dataset ,x_train , y_train ,x_test , y_test

def build_and_compile_cnn_model(K, M):

      
  
  
  
  # Build the model using the functional API
  # input layer
  i = Input(shape=M)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
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
  x = Dense(K, activation='softmax')(x)
  
  model = Model(i, x)
  
  # model description
  model.summary()

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model
