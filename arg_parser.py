import argparse

def create_parser():
    parser = argparse.ArgumentParser()

    # Weights & Biases configuration
    parser.add_argument('-wp', '--wandb_project', default='myprojectname', help="project name used to track experiments in weights & biases dashboard")
    parser.add_argument('-we', '--wandb_entity', default='myname', help="wandb entity used to track experiments in the weights & biases dashboard.")

    # Dataset configuration
    parser.add_argument('-d', '--dataset', choices=["mnist", "fashion_mnist"], default='fashion_mnist', help="dataset loader: choices = ['mnist', 'fashion_mnist']")

    # Training configuration
    parser.add_argument('-e', '--epochs', type=int, default=1, help="number of epochs to train neural network.")
    parser.add_argument('-b', '--batch_size', type=int, default=4, help="batch size used to train neural network.")

    # Loss and optimizer configuration
    parser.add_argument('-l', '--loss', choices=["mean_squared_error", "cross_entropy"], default='cross_entropy', help="loss function: choices = ['mean_squared_error', 'cross_entropy']")
    parser.add_argument('-o', '--optimizer', choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default='sgd', help="optimizer for backpropagation: choices = ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")

    # Learning rate and optimizer-specific parameters
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help="learning rate")
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help="momentum for momentum and nesterov optimizers")
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help="beta for rmsprop optimizer")
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help="beta1 for adam and nadam optimizers")
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help="beta2 for adam and nadam optimizers")
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001, help="epsilon for optimizers")
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help="weight decay for optimizers (L2 regularization)")

    # Weight initialization
    parser.add_argument('-w_i', '--weight_init', choices=["random", "xavier"], default='random', help="weight initialization method: choices = ['random', 'xavier']")

    # Neural network architecture
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help="number of hidden layers")
    parser.add_argument('-sz', '--hidden_size', type=list, default=[4], help="hidden layer size: [h1, h2...., hn]")
    parser.add_argument('-a', '--activation', choices=["identity", "sigmoid", "tanh", "ReLU"], default='sigmoid', help="activation function: choices = ['identity', 'sigmoid', 'tanh', 'ReLU']")

    return parser
