import numpy as np
import matplotlib.pyplot as plt

import neural_network as nn
from arg_parser import create_parser

from keras.utils import to_categorical
from keras.datasets import fashion_mnist, mnist

import wandb
from types import SimpleNamespace

###############################################

def visualize_data(x_train, y_train):    
    n_class = np.unique(y_train)

    _, axs = plt.subplots(10, 10, figsize=(10, 10))
    
    for i, d in enumerate(n_class):
        for j in range(len(n_class)):
            axs[j, i].imshow(x_train[y_train == d][j])
            axs[j, i].axis('off')         

    plt.show()
    return

def load_and_visualize_dataset(choice = "fashion_mnist"):

    if choice == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif choice == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        raise Exception("Invalid dataset.")    
    visualize_data(x_train, y_train)
    
    return x_train, y_train , x_test, y_test

def process_dataset(x_train, y_train, x_test, y_test):

    # reshape the images into n-dimensional vector
    x_train = x_train.reshape(x_train.shape[0], -1)    
    x_test = x_test.reshape(x_test.shape[0], -1)

    # normalize x_train and x_test pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # one-hot encode y_train and y_test values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train , x_test, y_test

if __name__=="__main__":
   
    parser = create_parser()
    args = vars(parser.parse_args())

    wandb.login(key='68d634810410e6fa1baf445fbc2ab0f586e0272e')

    sweep_config = {
    'method': 'grid',
    'name' : 'sweep cross entropy',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {'epochs': {'values': [5,10]},
                   'num_hidden_layers': {'values': [3,4,5]},
                   'hidden_size':{'values':[32,64,128]},
                   'weight_decay': {'values': [0, 0.0005, 0.5]},
                   'learning_rate': {'values': [1e-3, 1e-4]},
                   'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
                   'batch_size': {'values': [16,32,64]},
                   'weight_init': {'values': ['random','xavier']},
                   'activation': {'values': ['sigmoid','ReLU','tanh']},
                   'loss': {'values': ['cross_entropy']}
                   }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project='CS6910_Assignment_1')


    #args = {'wandb_project': 'myprojectname', 'wandb_entity': 'myname', 'dataset': 'fashion_mnist',
    #        'epochs': 1, 'batch_size': 4, 'loss': 'cross_entropy', 'optimizer': 'sgd', 'learning_rate': 0.1,
    #       'momentum': 0.5, 'beta': 0.5, 'beta1': 0.5, 'beta2': 0.5, 'epsilon': 1e-06, 'weight_decay': 0.0,
    #       'weight_init': 'random', 'num_layers': 1, 'hidden_size': 4, 'activation': 'sigmoid'}

    
    x_train, y_train , x_test, y_test = load_and_visualize_dataset(choice = args['dataset'])    
    x_train, y_train , x_test, y_test = process_dataset(x_train, y_train , x_test, y_test)

    with wandb.init() as run:

        run_name="-ac_"+wandb.config.activation+"-hs"+str(wandb.config.hidden_size)
        wandb.run.name=run_name
        
        model = nn.FeedForwardNN(input_size=x_train.shape[-1], num_hidden_layers=args['num_layers'],
                              hidden_size=args['hidden_size'], output_size=10, weight_initializer=args['weight_init'])
        model.train(args, x_train, y_train)
        model.predict(x_test, y_test, activationfn=args['activation'])

    wandb.agent(sweep_id, count=10) # calls main function for count number of times.
    wandb.finish()
    