# Assignment 1 - Feed Forward Neural Networks

This file contains the implementation of a FeedForward Neural Network (FFNN) in Python using NumPy. 

A feedforward neural network (FNN) is one of the two broad types of artificial neural network, characterized by direction of the flow of information between its layers.Its flow is uni-directional, meaning that the information in the model flows in only one direction—forward—from the input nodes, through the hidden nodes (if any) and to the output nodes, without any cycles or loops, in contrast to recurrent neural networks, which have a bi-directional flow. Modern feedforward networks are trained using the backpropagation method and are colloquially referred to as the "vanilla" neural networks. `Source: Wikipedia` 

The dataset to be used for training the network is MNIST or Fashion-MNIST.

### Structure

- **`FeedForwardNN` Class**: This class defines the structure and functionality of the FeedForward Neural Network.

- **Activation Functions**:
  - Identity
  - Sigmoid
  - Tanh
  - ReLU
  - Softmax

- **Loss Functions**:
  - Cross Entropy
  - Mean Squared Error

- **Optimizers**:
  - Stochastic Gradient Descent (SGD)
  - Momentum
  - Nesterov Accelerated Gradient (NAG)
  - RMSprop
  - Adam
  - NAdam

### Features

- **Initialization**: Provides options for weight initialization methods like random and Xavier initialization.
- **Forward Propagation**: Computes the forward pass of the network with different activation functions.
- **Backward Propagation**: Computes the backward pass of the network to update weights and biases using various optimizers.
- **Training**: Trains the network using provided training data with specified hyperparameters such as epochs, batch size, learning rate, etc.
- **Predictions**: Makes predictions on the test data and evaluates accuracy.
- **Optimizers**: Implements various optimization algorithms to improve training efficiency.
- **Accuracy Calculation**: Computes the accuracy of predictions.
- **Visualization**: Utilizes Weights and Biases (WandB) for visualization and tracking of training progress.

### Usage

To use this FFNN implementation:

1. Initialize the `FeedForwardNN` class with desired parameters such as input size, number of hidden layers, hidden layer size, output size, and weight initializer.

2. Choose activation functions, loss function, optimizer, and other hyperparameters for training.

3. Provide training data and call the `train` function.

4. Make predictions using the `predict` function.

### Requirements

- Python 3.x
- NumPy
- tqdm
- scikit-learn (for train-test split)
- Weights and Biases (WandB) for visualization (optional)

### Acknowledgments

This implementation is inspired by lecture slides of Prof. Mitesh M. Khapra on Deep Learning and various other resources and tutorials on neural networks and deep learning.

## Utility Files

### load_visualize_data.py

#### Description:
This code segment visualizes the Fashion MNIST dataset using Matplotlib and Weights & Biases (wandb) for logging the visualizations. The dataset consists of grayscale images of fashion items categorized into 10 classes.

#### Dependencies:
- wandb
- numpy
- matplotlib
- keras

#### Usage:
1. Ensure all dependencies are installed.
2. Run the script 'load_visualize_data.py'.

### argparser.py
This file contain the function to create an argument parser to read the command line arguments provided by the user.

### best_model_conf_matrix.py
This file is used to train the model using the best hyperparameters discovered from the hyperparameter sweep feature using wandb. It also plots and logs the confusion matrix to the wandb dashboard.

### dataloader.py 
This file is used to load the dataset according to user's choice and preprocess the dataset (i.e. reshape the images into a n-dimensional vector, normalizing pixel values, and one hot encoding the labels.)

### hyperparameters_sweep.py
This file is used to run a hyper-parameter sweep over the hyperparameters to find the best set of parameters on prediction accuracy on the validation set. It uses the sweep() method of wandb.

### load_visualize_data.py
This file is used to load and visualize 1 sample image from each class in the dataset.

### train.py
This is the main file which can be used by the evaluators to run the training on the neural network based on different command line arguments. 

