
import tensorflow as tf
from matplotlib import pyplot
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.applications import resnet


def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims


def cifar_dataset():
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()
  val_size = X_test.shape[0] // 10

  X_val = X_test[:val_size]
  y_val = y_test[:val_size]

  X_test= X_test[val_size:]
  y_test= y_test[val_size:]
  x_train = preprocess_image_input(X_train) 
  x_test = preprocess_image_input(X_test)
  x_val = preprocess_image_input(X_val)

 
  return x_train , y_train ,x_test , y_test ,x_val ,y_val



      
def feature_extraction(inputs):

    feature_extractor = resnet.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None)(inputs)

    return feature_extractor


def classifier(inputs):
    
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x


def final_model(inputs):

    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

    resnet_feature_extractor = feature_extraction(resize)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output


def build_and_compile_cnn_model():

  inputs = tf.keras.layers.Input(shape=(32,32,3))
  
  classification_output = final_model(inputs) 
  model = tf.keras.Model(inputs=inputs, outputs = classification_output)
  for layer in model.layers[:-1]:   #prepei na allaxtei analoga me poso upologistiki dunami exoume
    layer.trainable=False
    
  #for layer in model.layers[-4:]:
     # layer.trainable=True
  model.compile(optimizer='SGD', 
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])
  model.summary()
  return model



