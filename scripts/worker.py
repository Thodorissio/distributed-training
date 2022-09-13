import sys
import tensorflow as tf

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('/home/user/distributed-training/')
import models

if __name__ == '__main__':
    
    args = sys.argv[1:]

    nodes = int(args[0])
    model_name = args[1]

    if model_name == 'cifar_10':

        cifar_batch_size = 256 // nodes
        epochs = 5
        model = models.Cifar_10(batch_size = cifar_batch_size, epochs=epochs)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
            time, accuracy = model.run_model()

        print(f'time: {time}s')
        print(f'accuracy: {accuracy}')

    elif model_name  == 'bert_imdb':

        bert_batch_size = 512 // nodes
        epochs = 5
        model = models.IMDB_sentiment(batch_size = bert_batch_size, epochs=epochs)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
            time, accuracy = model.run_model()

        print(f'time: {time}s')
        print(f'accuracy: {accuracy}')

    elif model_name  == 'natural_images_densenet':

        densenet_batch_size = 512 // nodes
        epochs = 5
        model = models.Natural_images_densenet(batch_size=densenet_batch_size, epochs=epochs)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
            time, accuracy = model.run_model()

        print(f'time: {time}s')
        print(f'accuracy: {accuracy}')

    elif model_name  == 'fashion_mnist':

        fashion_mnist_batch_size = 512 // nodes
        epochs = 5
        model = models.Fashion_mnist(batch_size=fashion_mnist_batch_size, epochs=epochs)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
            time, accuracy = model.run_model()
            
        print(f'time: {time}s')
        print(f'accuracy: {accuracy}')

    elif model_name  == 'mnist':

        mnist_batch_size = 512 // nodes
        epochs = 5
        model = models.Mnist_restnet(batch_size=mnist_batch_size, epochs=epochs)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
            time, accuracy = model.run_model()

        print(f'time: {time}s')
        print(f'accuracy: {accuracy}')

    else:
        
        print('Invalid dataset given.')
        print('Availiable datasets:')
        print('- cifar_10')
        print('- bert_imdb')
        print('- natural_images_densenet')
        print('- fashion_mnist')
        print('- mnist')