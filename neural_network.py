import numpy as np
class FeedForwardNN:
    def __init__(self, input_size:int, num_hidden_layers:int, hidden_size:list, output_size:int, weight_initializer):
        
        self.in_dims = input_size             # size of the input layer: 784      
        self.n_hidden = num_hidden_layers     # num of hidden layers in the network (excluding output layer)
        self.h_dims = hidden_size             # size of the hidden layers: list of integers [h1, h2, ....]
        self.out_dims = output_size           # size of the output layer: 10       

        self.layers = self.n_hidden + 1        

        self.parameters = self.weight_init(weight_initializer)
    
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
    
    def d_softmax(self, x):
        softmax_output = self.softmax(x)
        num_classes = softmax_output.shape[1]
        d_softmax = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    d_softmax[i, j] = softmax_output[i] * (1 - softmax_output[i])
                else:
                    d_softmax[i, j] = -softmax_output[i] * softmax_output[j]
        
        return d_softmax


    
    #########################################################################################################################   
    #### LOSS FUNCTIONS

    def cross_entropy(self, y, yhat):
        return -np.sum(y * np.log(yhat + 1e-8))
    
    def mean_squared_error(self, y, yhat):
        return np.sum(np.square(y - yhat))
            
    #########################################################################################################################
    #### INITIALIZE PARAMETERS

    def xavier_init(self, shape:tuple):         
        dim_in, dim_out = shape[0], shape[1]
        limit = np.sqrt(6 / (dim_in + dim_out))
        return np.random.uniform(-limit, limit, size=shape)
    
    def random_init(self, shape:tuple):        
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
            raise Exception("NotImplementedError: Invalid Initializer Method.")

        return params
    #########################################################################################################################
    #### FORWARD PROPAGATION

    def forward_propagation(self, X, activ_fn = 'sigmoid'):
        self.activations = {}
        self.activations["A0"] = X
        self.activations["H0"] = X

        if activ_fn == 'sigmoid':
            for i in range(1, self.layers):
                self.activations["A"+str(i)] = np.dot(self.activations["H"+str(i-1)],self.parameters["W"+str(i)]) + self.parameters["b"+str(i)]
                self.activations["H"+str(i)] = self.sigmoid(self.activations["A"+str(i)])

        elif activ_fn == 'identity':
            for i in range(1, self.layers):
                self.activations["A"+str(i)] = np.dot(self.activations["H"+str(i-1)],self.parameters["W"+str(i)]) + self.parameters["b"+str(i)]
                self.activations["H"+str(i)] = self.identity(self.activations["A"+str(i)])

        elif activ_fn == 'tanh':
            for i in range(1, self.layers):
                self.activations["A"+str(i)] = np.dot(self.activations["H"+str(i-1)],self.parameters["W"+str(i)]) + self.parameters["b"+str(i)]
                self.activations["H"+str(i)] = self.tanh(self.activations["A"+str(i)])

        elif activ_fn == 'ReLU':
            for i in range(1, self.layers):
                self.activations["A"+str(i)] = np.dot(self.activations["H"+str(i-1)],self.parameters["W"+str(i)]) + self.parameters["b"+str(i)]
                self.activations["H"+str(i)] = self.relu(self.activations["A"+str(i)])

        else:
            raise Exception("NotImplementedError: Invalid Activation Function.")
        
        self.activations["A"+str(self.layers)] = np.dot(self.activations["H"+str(self.layers-1)], self.parameters["W"+str(self.layers)]) + self.parameters["b"+str(self.layers)]        
        self.Y_hat = self.softmax(self.activations["A"+str(self.layers)])        
        
        return self.Y_hat
   
    #########################################################################################################################
    #### BACKWARD PROPAGATION

    def backward_propagation(self, X,  Y, optimizer = 'sgd', lr=0.1, momentum=0.5, beta=0.5, beta1=0.5, beta2=0.5, epsilon=0.000001, loss_fn = 'cross_entropy', activ_fn = 'sigmoid'):
        grad = {}
        N = X.shape[0]

        # Compute loss w.r.t. the given loss function
        if loss_fn == 'cross_entropy':
            loss = self.cross_entropy(Y, self.Y_hat) / N
        else:
            loss = self.mean_squared_error(Y, self.Y_hat) / N
                
        # Compute gradient w.r.t. the output
        grad['dA'+str(self.layers)] = self.Y_hat - Y  # Gradient with respect to Softmax loss       

        for k in range(self.layers, 0, -1):
            # Compute the gradients w.r.t. parameters W and b
            grad['dW'+str(k)] = np.dot(np.transpose(self.activations["H"+str(k-1)]),grad["dA"+str(k)])
            grad['db'+str(k)] = grad["dA"+str(k)]

            # Compute the gradients w.r.t. activation (H[k-1]) in the layer below
            grad['dH'+str(k-1)] = np.dot(grad["dA"+str(k)], np.transpose(self.parameters['W'+str(k)]))

            # Compute the gradients w.r.t. pre-activations (A[k-1]) in the layer below
            if activ_fn == 'sigmoid':
                grad['dA'+str(k-1)] = np.multiply(grad['dH'+str(k-1)],self.d_sigmoid(self.activations['A'+str(k-1)]))
            elif activ_fn == 'identity':
                grad['dA'+str(k-1)] = np.multiply(grad['dH'+str(k-1)],self.d_identity(self.activations['A'+str(k-1)]))
            elif activ_fn == 'tanh':
                grad['dA'+str(k-1)] = np.multiply(grad['dH'+str(k-1)],self.d_tanh(self.activations['A'+str(k-1)]))
            elif activ_fn == 'ReLU':
                grad['dA'+str(k-1)] = np.multiply(grad['dH'+str(k-1)],self.d_relu(self.activations['A'+str(k-1)]))
            else:
                raise Exception("NotImplementedError: Invalid Activation Function.")

        if optimizer == 'sgd':
            self.sgd_update(grad, lr)
        elif optimizer == 'momentum':
            self.momentum_update(grad, lr, momentum)
        elif optimizer == 'nesterov':
            self.nesterov_update(grad, lr, momentum)
        elif optimizer == 'rmsprop':
            self.rmsprop_update(grad, lr, beta, epsilon)
        elif optimizer == 'adam':
            self.adam_update(grad, lr, beta1, beta2, epsilon)
        elif optimizer == 'nadam':
            self.nadam_update(grad, lr, beta1, beta2, epsilon)
    
        return loss

    #########################################################################################################################
    #### OPTIMIZERS
        
    def sgd_update(self, grad, lr):
        for i in range(1, self.layers + 1):
            self.parameters['W'+str(i)] -= lr * grad['dW'+str(i)]
            self.parameters['b'+str(i)] -= lr * np.sum(grad['db'+str(i)], axis=0, keepdims=True)

    def momentum_update(self, grad, lr, momentum):
        self.V = {}
        for i in range(1, self.layers + 1):
            if 'v_dW'+str(i) not in self.V:
                self.V['v_dW'+str(i)] = np.zeros_like(self.parameters['W'+str(i)])
                self.V['v_db'+str(i)] = np.zeros_like(self.parameters['b'+str(i)])

            self.V['v_dW'+str(i)] = momentum * self.V['v_dW'+str(i)] + grad['dW'+str(i)]
            self.V['v_db'+str(i)] = momentum * self.V['v_db'+str(i)] + np.sum(grad['db'+str(i)], axis=0, keepdims=True)

            self.parameters['W'+str(i)] -= lr * self.V['v_dW'+str(i)]
            self.parameters['b'+str(i)] -= lr * self.V['v_db'+str(i)]

    def nesterov_update(self, grad, lr, momentum):
        self.V = {}
        for i in range(1, self.layers + 1):
            if 'v_dW'+str(i) not in self.V:
                self.V['v_dW'+str(i)] = np.zeros_like(self.parameters['W'+str(i)])
                self.V['v_db'+str(i)] = np.zeros_like(self.parameters['b'+str(i)])
            
            self.V['v_dW'+str(i)] = momentum * self.V['v_dW'+str(i)] - lr * grad['dW'+str(i)]
            self.V['v_db'+str(i)] = momentum * self.V['v_db'+str(i)] - lr * np.sum(grad['db'+str(i)], axis=0, keepdims=True)
            
            self.parameters['W'+str(i)] += self.V['v_dW'+str(i)]
            self.parameters['b'+str(i)] += self.V['v_db'+str(i)]


    def rmsprop_update(self, grad, lr, beta, epsilon):
        self.V = {}
        for i in range(1, self.layers + 1):
            if 's_dW'+str(i) not in self.V:
                self.V['s_dW'+str(i)] = np.zeros_like(self.parameters['W'+str(i)])
                self.V['s_db'+str(i)] = np.zeros_like(self.parameters['b'+str(i)])

            self.V['s_dW'+str(i)] = beta * self.V['s_dW'+str(i)] + (1 - beta) * np.square(grad['dW'+str(i)])
            self.V['s_db'+str(i)] = beta * self.V['s_db'+str(i)] + (1 - beta) * np.square(grad['db'+str(i)])

            self.parameters['W'+str(i)] -= lr * grad['dW'+str(i)] / (np.sqrt(self.V['s_dW'+str(i)]) + epsilon)
            self.parameters['b'+str(i)] -= lr * np.sum(grad['db'+str(i)], axis=0, keepdims=True) / (np.sqrt(self.V['s_db'+str(i)]) + epsilon)

    def adam_update(self, grad, lr, beta1, beta2, time_step, epsilon):        
        self.V = {}
        for i in range(1, self.layers + 1):
            if 'm_dW'+str(i) not in self.V:
                self.V['m_dW'+str(i)] = np.zeros_like(self.parameters['W'+str(i)])
                self.V['m_db'+str(i)] = np.zeros_like(self.parameters['b'+str(i)])
                self.V['v_dW'+str(i)] = np.zeros_like(self.parameters['W'+str(i)])
                self.V['v_db'+str(i)] = np.zeros_like(self.parameters['b'+str(i)])

            self.V['m_dW'+str(i)] = beta1 * self.V['m_dW'+str(i)] + (1 - beta1) * grad['dW'+str(i)]
            self.V['m_db'+str(i)] = beta1 * self.V['m_db'+str(i)] + (1 - beta1) * grad['db'+str(i)]
            self.V['v_dW'+str(i)] = beta2 * self.V['v_dW'+str(i)] + (1 - beta2) * np.square(grad['dW'+str(i)])
            self.V['v_db'+str(i)] = beta2 * self.V['v_db'+str(i)] + (1 - beta2) * np.square(grad['db'+str(i)])

            m_dW_hat = self.V['m_dW'+str(i)] / (1 - beta1 ** time_step)
            m_db_hat = self.V['m_db'+str(i)] / (1 - beta1 ** time_step)
            v_dW_hat = self.V['v_dW'+str(i)] / (1 - beta2 ** time_step)
            v_db_hat = self.V['v_db'+str(i)] / (1 - beta2 ** time_step)

            self.parameters['W'+str(i)] -= lr * m_dW_hat / (np.sqrt(v_dW_hat) + epsilon)
            self.parameters['b'+str(i)] -= lr * m_db_hat / (np.sqrt(v_db_hat) + epsilon)

    def nadam_update(self, grad, lr, beta1, beta2, time_step, epsilon):
        self.V = {}
        for i in range(1, self.layers + 1):
            if 'm_dW'+str(i) not in self.V:
                self.V['m_dW'+str(i)] = np.zeros_like(self.parameters['W'+str(i)])
                self.V['m_db'+str(i)] = np.zeros_like(self.parameters['b'+str(i)])
                self.V['v_dW'+str(i)] = np.zeros_like(self.parameters['W'+str(i)])
                self.V['v_db'+str(i)] = np.zeros_like(self.parameters['b'+str(i)])

            self.V['m_dW'+str(i)] = beta1 * self.V['m_dW'+str(i)] + (1 - beta1) * grad['dW'+str(i)]
            self.V['m_db'+str(i)] = beta1 * self.V['m_db'+str(i)] + (1 - beta1) * grad['db'+str(i)]
            self.V['v_dW'+str(i)] = beta2 * self.V['v_dW'+str(i)] + (1 - beta2) * np.square(grad['dW'+str(i)])
            self.V['v_db'+str(i)] = beta2 * self.V['v_db'+str(i)] + (1 - beta2) * np.square(grad['db'+str(i)])

            m_dW_hat = self.V['m_dW'+str(i)] / (1 - beta1 ** time_step)
            m_db_hat = self.V['m_db'+str(i)] / (1 - beta1 ** time_step)
            v_dW_hat = self.V['v_dW'+str(i)] / (1 - beta2 ** time_step)
            v_db_hat = self.V['v_db'+str(i)] / (1 - beta2 ** time_step)

            self.parameters['W'+str(i)] -= lr * (beta1 * m_dW_hat + (1 - beta1) * grad['dW'+str(i)]) / (np.sqrt(v_dW_hat) + epsilon)
            self.parameters['b'+str(i)] -= lr * (beta1 * m_db_hat + (1 - beta1) * grad['db'+str(i)]) / (np.sqrt(v_db_hat) + epsilon)
    
    #########################################################################################################################
    #### TRAIN FUNCTION
        
    def train(self, args, X_train, Y_train):
        """
        Extract the arguments passed from the CLI.

        args = {'wandb_project': 'myprojectname', 'wandb_entity': 'myname', 'dataset': 'fashion_mnist',
                'epochs': 1, 'batch_size': 4, 'loss': 'cross_entropy', 'optimizer': 'sgd', 'learning_rate': 0.1,
                'momentum': 0.5, 'beta': 0.5, 'beta1': 0.5, 'beta2': 0.5, 'epsilon': 1e-06, 'weight_decay': 0.0,
                'weight_init': 'random', 'num_layers': 1, 'hidden_size': 4, 'activation': 'sigmoid'}

        """
        epochs = args['epochs']
        batch_size = args['batch_size']
        lossfn = args['loss']
        optimizer = args['optimizer']
        lr = args['learning_rate']
        momentum = args['momentum']
        beta = args['beta']
        beta1 = args['beta1']
        beta2 = args['beta2']
        eps = args['epsilon']
        weight_decay = args['weight_decay']        
        activationfn = args['activation']


        num_samples = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle the training data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
        
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                # Forward propagation
                x_batch = X_train[batch_indices]
                y_batch = Y_train[batch_indices]

                y_pred = self.forward_propagation(x_batch, activ_fn=activationfn)

                # Calculate accuracy
                accuracy = self.calculate_accuracy(y_batch, y_pred)

                # Backward propagation
                loss = self.backward_propagation(x_batch, y_batch, optimizer = optimizer, lr=lr, momentum=momentum,
                                                 beta=beta, beta1=beta1, beta2=beta2, epsilon=eps,
                                                 loss_fn = lossfn, activ_fn = activationfn)

            if epoch % 100 == 0:
                print(f"Training: Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    #########################################################################################################################
    #### CALCULATE ACCURACY
    
    def calculate_accuracy(self, y_true, y_pred):
        correct = 0
        for i in range(len(y_true)):
            if np.argmax(y_true[i]) == np.argmax(y_pred[i]):
                correct += 1

        accuracy = correct / len(y_true)
        return accuracy
    
    #########################################################################################################################
    #### PREDICTIONS

    def predict(self, x_test, y_test, activationfn):        

        y_pred = self.forward_propagation(x_test, activ_fn=activationfn)
        accuracy = self.calculate_accuracy(y_test, y_pred)

        print(f"Test Accuracy: {accuracy:.4f}")