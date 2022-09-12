import sys
import tensorflow as tf

sys.path.append('../')
import models

if __name__ == '__main__':
    
    args = sys.argv[1:]
    
    nodes = int(args[0])
    model_name = args[1]

    if model_name == 'cifar_10':
        cifar_batch_size = 512 // nodes
        model = models.Cifar_10(batch_size = cifar_batch_size)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            time, accuracy = model.run_model()
        print(f'time: {time}s')
        print(f'accuracy: {accuracy}')
    else:
        print('Invalid dataset given.')
        print('Availiable datasets:')
        print('- cifar_10')
        print('- imdb_sentiment')
        print('- mnist')
        print('- fashion_mnist')

