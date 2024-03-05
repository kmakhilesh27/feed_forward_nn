import wandb
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FeedForwardNN:
    def __init__(self, input_size:int, num_hidden_layers:int, hidden_size:list, output_size:int):
        
        self.in_dims = input_size             # size of the input layer: 784      
        self.n_hidden = num_hidden_layers     # num of hidden layers in the network (excluding output layer)
        self.h_dims = hidden_size             # size of the hidden layers: list of integers [h1, h2, ....]
        self.out_dims = output_size           # size of the output layer: 10       

        self.layers = self.n_hidden + 1        

        self.parameters = self.weight_init()
    
    #########################################################################################################################
    #### ACTIVATION FUNCTIONS
    
    def identity(self, x):
        return x
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))    
    
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    #### DERIVATIVES

    def d_identity(self, x):
        return np.ones_like(x)
        
    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def d_tanh(self, x):
        return 1 - np.square(self.tanh(x))
    
    def d_relu(self, x):
        return np.where(x > 0, 1, 0)

    

    #########################################################################################################################   
    #### LOSS FUNCTIONS

    def cross_entropy(self, y, yhat):
        return -np.sum(y * np.log(yhat + 1e-8))
    
    def mean_squared_error(self, y, yhat):
        return np.sum(np.square(y - yhat))
            
    #########################################################################################################################
    #### INITIALIZE PARAMETERS

    def xavier_init(self, shape:tuple):
        # xavier_init((output_size, input_size)) 
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)
    
    def random_init(self, shape:tuple):
        # random_init((output_size, input_size))
        return np.random.randn(*shape)

    def weight_init(self, choice = "random"):
        params = {}
        if choice == "random":
            # parameters for the first hidden layer
            params["W1"] = self.random_init((self.in_dims, self.h_dims[0]))
            params["b1"] = np.zeros((1, self.h_dims[0]))
            # parameters from the second hidden layer to last hidden layer
            for i in range(1,self.layers - 1):
                params["W"+str(i+1)] = self.random_init((self.h_dims[i-1],self.h_dims[i]))
                params["b"+str(i+1)] = np.zeros((1, self.h_dims[i]))
            # parameters for the output layer
            params["W"+str(self.layers)] = self.random_init((self.h_dims[-1], self.out_dims))
            params["b"+str(self.layers)] = np.zeros((1, self.out_dims))
        elif choice == "xavier":
            # parameters for the first hidden layer
            params["W1"] = self.xavier_init((self.in_dims, self.h_dims[0]))
            params["b1"] = np.zeros((1, self.h_dims[0]))
            # parameters from the second hidden layer to last hidden layer
            for i in range(1,self.layers - 1):
                params["W"+str(i+1)] = self.xavier_init((self.h_dims[i-1],self.h_dims[i]))
                params["b"+str(i+1)] = np.zeros((1, self.h_dims[i]))
            # parameters for the output layer
            params["W"+str(self.layers)] = self.xavier_init((self.h_dims[-1], self.out_dims))
            params["b"+str(self.layers)] = np.zeros((1, self.out_dims))
        
        else:
            raise Exception("Invalid initialization method.")

        return params
    #########################################################################################################################
    #### FORWARD PROPAGATION

    def forward_propagation(self, X):
        activations = {}
        
        activations["A1"] = np.dot(X, self.parameters["W1"]) + self.parameters["b1"]
        activations["H1"] = self.sigmoid(activations["A1"])

        for i in range(2, self.layers):
            activations["A"+str(i)] = np.dot(activations["H"+str(i-1)],self.parameters["W"+str(i)]) + self.parameters["b"+str(i)]
            activations["H"+str(i)] = self.sigmoid(activations["A"+str(i)])
        
        activations["A"+str(self.layers)] = np.dot(activations["H"+str(self.layers-1)], self.parameters["W"+str(self.layers)]) + self.parameters["b"+str(self.layers)]
        activations["H"+str(self.layers)] = self.softmax(activations["A"+str(self.layers)])

        return activations    
    #########################################################################################################################
    #### BACKWARD PROPAGATION

    def backward_propagation(self, Y, activations):
        grads = {}
                
        # Compute gradient w.r.t. the output
        grads['dA'+str(self.layers)] = -(Y - activations['H'+str(self.layers)])

        for k in range(self.layers, 1, -1):
            # Compute the gradients w.r.t. parameters W and b
            grads['dW'+str(k)] = np.dot(np.transpose(activations["H"+str(k-1)]),grads["dA"+str(k)])
            grads['db'+str(k)] = grads["dA"+str(k)]

            # Compute the gradients w.r.t. activation (H[k-1]) in the layer below
            grads['dH'+str(k-1)] = np.dot(grads["dA"+str(k)], np.transpose(self.parameters['W'+str(k)]))

            # Compute the gradients w.r.t. pre-activations (A[k-1]) in the layer below
            grads['dA'+str(k-1)] = np.multiply(grads['dH'+str(k-1)],self.sigmoid_d(activations['A'+str(k-1)])) 

        return grads
    #########################################################################################################################
    #### OPTIMIZERS
    
    def update_parameters(self, grads, learning_rate):
        
        for k in range(1, self.layers + 1):
            self.parameters["W"+str(k)] = self.parameters["W"+str(k)] - learning_rate * grads["dW"+str(k)]
            self.parameters["b"+str(k)] = self.parameters["b"+str(k)] - learning_rate * grads["db"+str(k)]

        return
 
    def train(self, X_train, Y_train, epoch=1000, learning_rate=0.001, minibatch_size=64):
        pass