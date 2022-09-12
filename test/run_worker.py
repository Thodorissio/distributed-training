import sys

sys.path.append('../')
import models

if __name__ == '__main__':
    
    args = sys.argv[1:]
    
    nodes = args[0]
    model_name = args[1]

    if model_name == 'cifar_10':
        cifar_batch_size = 512 // nodes
        model = models.Cifar_10(batch = cifar_batch_size)
        time, accuracy = model.run_model()
    else:
        print('Invalid dataset given.')
        print('Availiable datasets:')
        print('- cifar_10')
        print('- imdb_sentiment')
        print('- mnist')
        print('- fashion_mnist')

