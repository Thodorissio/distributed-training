import sys
import pandas as pd
import tensorflow as tf
from os.path import exists

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('/home/user/distributed-training/')
import models

if __name__ == '__main__':
    
    args = sys.argv[1:]

    nodes = int(args[0])
    model_name = args[1]

    print(args)

    save_flag = True

    if model_name == 'cifar_10':

        batch_size = 256 // nodes
        epochs = 5
        model = models.Cifar_10(batch_size = batch_size, epochs=epochs)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
            time, accuracy = model.run_model()      

    elif model_name  == 'bert_imdb':

        batch_size = 512 // nodes
        epochs = 5
        model = models.IMDB_sentiment(batch_size = batch_size, epochs=epochs)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
            time, accuracy = model.run_model()      

    elif model_name  == 'natural_images_densenet':

        batch_size = 512 // nodes
        epochs = 5
        model = models.Natural_images_densenet(batch_size=batch_size, epochs=epochs)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
            time, accuracy = model.run_model()      

    elif model_name  == 'fashion_mnist':

        fashion_mnist_batch_size = 512 // nodes
        epochs = 5
        model = models.Fashion_mnist(batch_size=fashion_mnist_batch_size, epochs=epochs)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
            time, accuracy = model.run_model()
              

    elif model_name  == 'mnist':

        batch_size = 512 // nodes
        epochs = 5
        model = models.Mnist_restnet(batch_size=batch_size, epochs=epochs)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
            time, accuracy = model.run_model()      

    else:
        
        save_flag = False
        print('Invalid dataset given.')
        print('Availiable datasets:')
        print('- cifar_10')
        print('- bert_imdb')
        print('- natural_images_densenet')
        print('- fashion_mnist')
        print('- mnist')
    
    if save_flag and len(args) > 2:

        res_df = pd.DataFrame({
            'model': model_name, 'nodes': nodes, 'training_time': time,
            'training_accuracy': accuracy, 'epochs': epochs, 'batch_size': batch_size}, index=[0])

        if exists('/home/user/results.csv'):
            res_df.to_csv('/home/user/results.csv', mode='a', index=False, header=False)
        else:
            res_df.to_csv('/home/user/results.csv', index=False)