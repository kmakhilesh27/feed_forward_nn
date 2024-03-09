import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tq
from tqdm.notebook import tqdm_notebook as tqnb

# Set the random seed for numpy for reproducibility
np.random.seed(42)
 
class FeedForwardNN:
    
    def __init__(self, input_size:int, num_hidden_layers:int, hidden_size:list, output_size:int, weight_initializer):
        
        self.in_dims = input_size             # size of the input layer: 784      
        self.n_hidden = num_hidden_layers     # num of hidden layers in the network (excluding output layer)
        self.h_dims = hidden_size             # size of the hidden layers: list of integers [h1, h2, ....]
        self.out_dims = output_size           # size of the output layer: 10       

        self.layers = self.n_hidden + 1        

        self.parameters = self.parameter_initializer(weight_initializer)
    
    #########################################################################################################################
    '''ACTIVATION FUNCTIONS'''
    
    def _identity(self, x):
        return x
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))    
    
    def _tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    '''DERIVATIVES'''

    def _d_identity(self, x):
        return np.ones_like(x)
        
    def _d_sigmoid(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))
    
    def _d_tanh(self, x):
        return 1 - np.square(self._tanh(x))
    
    def _d_relu(self, x):
        return np.where(x > 0, 1, 0)
    
    def _d_softmax(self, x):
        softmax_output = self._softmax(x)
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
    '''LOSS FUNCTIONS'''

    def _cross_entropy(self, y, yhat):
        return -np.sum(y * np.log(yhat + 1e-8))
    
    def _mean_squared_error(self, y, yhat):
        return np.sum(np.square(y - yhat))
            
    #########################################################################################################################
    '''INITIALIZE PARAMETERS'''

    def _xavier_init(self, shape:tuple):         
        dim_in, dim_out = shape[0], shape[1]
        limit = np.sqrt(6 / (dim_in + dim_out))
        return np.random.uniform(-limit, limit, size=shape)
    
    def _random_init(self, shape:tuple):        
        return np.random.randn(*shape)

    def parameter_initializer(self, choice = "random"):
        params = {}
        if choice == "random":
            # parameters for the first hidden layer
            params["W1"] = self._random_init((self.in_dims, self.h_dims[0]))
            params["b1"] = np.zeros((1, self.h_dims[0]))
            # parameters from the second hidden layer to last hidden layer
            for i in range(1,self.layers - 1):
                params["W"+str(i+1)] = self._random_init((self.h_dims[i-1],self.h_dims[i]))
                params["b"+str(i+1)] = np.zeros((1, self.h_dims[i]))
            # parameters for the output layer
            params["W"+str(self.layers)] = self._random_init((self.h_dims[-1], self.out_dims))
            params["b"+str(self.layers)] = np.zeros((1, self.out_dims))
        elif choice == "xavier" or choice == "Xavier":
            # parameters for the first hidden layer
            params["W1"] = self._xavier_init((self.in_dims, self.h_dims[0]))
            params["b1"] = np.zeros((1, self.h_dims[0]))
            # parameters from the second hidden layer to last hidden layer
            for i in range(1,self.layers - 1):
                params["W"+str(i+1)] = self._xavier_init((self.h_dims[i-1],self.h_dims[i]))
                params["b"+str(i+1)] = np.zeros((1, self.h_dims[i]))
            # parameters for the output layer
            params["W"+str(self.layers)] = self._xavier_init((self.h_dims[-1], self.out_dims))
            params["b"+str(self.layers)] = np.zeros((1, self.out_dims))
        
        else:
            raise Exception("NotImplementedError: Invalid Initializer Method.")

        return params
    #########################################################################################################################
    '''FORWARD PROPAGATION'''

    def forward_propagation(self, X, activ_fn = 'sigmoid'):
        self.activations = {}
        self.activations["A0"] = X
        self.activations["H0"] = X

        if activ_fn == 'sigmoid':
            for i in range(1, self.layers):
                self.activations["A"+str(i)] = np.dot(self.activations["H"+str(i-1)],self.parameters["W"+str(i)]) + self.parameters["b"+str(i)]
                self.activations["H"+str(i)] = self._sigmoid(self.activations["A"+str(i)])

        elif activ_fn == 'identity':
            for i in range(1, self.layers):
                self.activations["A"+str(i)] = np.dot(self.activations["H"+str(i-1)],self.parameters["W"+str(i)]) + self.parameters["b"+str(i)]
                self.activations["H"+str(i)] = self._identity(self.activations["A"+str(i)])

        elif activ_fn == 'tanh':
            for i in range(1, self.layers):
                self.activations["A"+str(i)] = np.dot(self.activations["H"+str(i-1)],self.parameters["W"+str(i)]) + self.parameters["b"+str(i)]
                self.activations["H"+str(i)] = self._tanh(self.activations["A"+str(i)])

        elif activ_fn == 'ReLU':
            for i in range(1, self.layers):
                self.activations["A"+str(i)] = np.dot(self.activations["H"+str(i-1)],self.parameters["W"+str(i)]) + self.parameters["b"+str(i)]
                self.activations["H"+str(i)] = self._relu(self.activations["A"+str(i)])

        else:
            raise Exception("NotImplementedError: Invalid Activation Function.")
        
        self.activations["A"+str(self.layers)] = np.dot(self.activations["H"+str(self.layers-1)], self.parameters["W"+str(self.layers)]) + self.parameters["b"+str(self.layers)]        
        self.y_hat = self._softmax(self.activations["A"+str(self.layers)])        
        
        return self.y_hat
   
    #########################################################################################################################
    '''BACKWARD PROPAGATION'''

    def backward_propagation(self, epoch, X,  Y, optimizer = 'sgd', lr=0.1, momentum=0.5, beta=0.5, beta1=0.5, beta2=0.5, epsilon=0.000001, loss_fn = 'cross_entropy', activ_fn = 'sigmoid', no_grad=False):
        grad = {}
        num_samples = len(X)

        # Compute loss w.r.t. the given loss function
        if loss_fn == 'cross_entropy':
            loss = self._cross_entropy(Y, self.y_hat) / num_samples
        else:
            loss = self._mean_squared_error(Y, self.y_hat) / num_samples
        
        if no_grad == True:
            return loss
                
        # Compute gradient w.r.t. the output (with respect to Softmax loss)
        grad['dA'+str(self.layers)] = self.y_hat - Y  

        for k in range(self.layers, 0, -1):
            # Compute the gradients w.r.t. parameters W and b
            grad['dW'+str(k)] = np.dot(np.transpose(self.activations["H"+str(k-1)]),grad["dA"+str(k)])
            grad['db'+str(k)] = grad["dA"+str(k)]

            # Compute the gradients w.r.t. activation (H[k-1]) in the layer below
            grad['dH'+str(k-1)] = np.dot(grad["dA"+str(k)], np.transpose(self.parameters['W'+str(k)]))

            # Compute the gradients w.r.t. pre-activations (A[k-1]) in the layer below
            if activ_fn == 'sigmoid':
                grad['dA'+str(k-1)] = np.multiply(grad['dH'+str(k-1)],self._d_sigmoid(self.activations['A'+str(k-1)]))
            elif activ_fn == 'identity':
                grad['dA'+str(k-1)] = np.multiply(grad['dH'+str(k-1)],self._d_identity(self.activations['A'+str(k-1)]))
            elif activ_fn == 'tanh':
                grad['dA'+str(k-1)] = np.multiply(grad['dH'+str(k-1)],self._d_tanh(self.activations['A'+str(k-1)]))
            elif activ_fn == 'ReLU':
                grad['dA'+str(k-1)] = np.multiply(grad['dH'+str(k-1)],self._d_relu(self.activations['A'+str(k-1)]))
            else:
                raise Exception("NotImplementedError: Invalid Activation Function.")
        
        # Call the optimizer according to the passed argument to update W and b values
        if optimizer == 'sgd':
            self._sgd_update(grad, lr)
        elif optimizer == 'momentum':
            self._momentum_update(grad, lr, momentum)
        elif optimizer == 'nesterov':
            self._nesterov_update(grad, lr, momentum)
        elif optimizer == 'rmsprop':
            self._rmsprop_update(grad, lr, beta, epsilon)
        elif optimizer == 'adam':
            self._adam_update(grad, lr, beta1, beta2, epoch, epsilon)
        elif optimizer == 'nadam':
            self._nadam_update(grad, lr, beta1, beta2, epoch, epsilon)
    
        return loss

    #########################################################################################################################
    '''OPTIMIZERS'''
        
    def _sgd_update(self, grad, lr):
        for i in range(1, self.layers + 1):
            self.parameters['W'+str(i)] -= lr * grad['dW'+str(i)]
            self.parameters['b'+str(i)] -= lr * np.sum(grad['db'+str(i)], axis=0, keepdims=True)

    def _momentum_update(self, grad, lr, momentum):
        self.V = {}
        for i in range(1, self.layers + 1):
            if 'v_dW'+str(i) not in self.V:
                self.V['v_dW'+str(i)] = np.zeros_like(self.parameters['W'+str(i)])
                self.V['v_db'+str(i)] = np.zeros_like(self.parameters['b'+str(i)])

            self.V['v_dW'+str(i)] = momentum * self.V['v_dW'+str(i)] + grad['dW'+str(i)]
            self.V['v_db'+str(i)] = momentum * self.V['v_db'+str(i)] + np.sum(grad['db'+str(i)], axis=0, keepdims=True)

            self.parameters['W'+str(i)] -= lr * self.V['v_dW'+str(i)]
            self.parameters['b'+str(i)] -= np.sum(lr * self.V['v_db'+str(i)], axis = 0, keepdims=True)

    def _nesterov_update(self, grad, lr, momentum):
        self.V = {}
        for i in range(1, self.layers + 1):
            if 'v_dW'+str(i) not in self.V:
                self.V['v_dW'+str(i)] = np.zeros_like(self.parameters['W'+str(i)])
                self.V['v_db'+str(i)] = np.zeros_like(self.parameters['b'+str(i)])
            
            self.V['v_dW'+str(i)] = momentum * self.V['v_dW'+str(i)] - lr * grad['dW'+str(i)]
            self.V['v_db'+str(i)] = momentum * self.V['v_db'+str(i)] - lr * np.sum(grad['db'+str(i)], axis=0, keepdims=True)
            
            self.parameters['W'+str(i)] += self.V['v_dW'+str(i)]
            self.parameters['b'+str(i)] += np.sum(self.V['v_db'+str(i)], axis = 0, keepdims=True)


    def _rmsprop_update(self, grad, lr, beta, epsilon):
        self.V = {}
        for i in range(1, self.layers + 1):
            if 's_dW'+str(i) not in self.V:
                self.V['s_dW'+str(i)] = np.zeros_like(self.parameters['W'+str(i)])
                self.V['s_db'+str(i)] = np.zeros_like(self.parameters['b'+str(i)])

            self.V['s_dW'+str(i)] = beta * self.V['s_dW'+str(i)] + (1 - beta) * np.square(grad['dW'+str(i)])
            self.V['s_db'+str(i)] = beta * self.V['s_db'+str(i)] + (1 - beta) * np.square(grad['db'+str(i)])

            self.parameters['W'+str(i)] -= lr * grad['dW'+str(i)] / (np.sqrt(self.V['s_dW'+str(i)]) + epsilon)
            self.parameters['b'+str(i)] -= lr * np.sum(grad['db'+str(i)], axis=0, keepdims=True) / (np.sqrt(self.V['s_db'+str(i)]) + epsilon)

    def _adam_update(self, grad, lr, beta1, beta2, time_step, epsilon):        
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
            self.parameters['b'+str(i)] -= np.sum(lr * m_db_hat / (np.sqrt(v_db_hat) + epsilon), axis = 0, keepdims=True)

    def _nadam_update(self, grad, lr, beta1, beta2, time_step, epsilon):
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
            self.parameters['b'+str(i)] -= np.sum(lr * (beta1 * m_db_hat + (1 - beta1) * grad['db'+str(i)]), axis = 0, keepdims=True) / (np.sqrt(v_db_hat) + epsilon)
    
    #########################################################################################################################
    '''TRAIN FUNCTION'''
        
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

        # Lists to accumulate the losses over each epoch (size: num_epochs)
        self.training_loss = [] 
        self.validation_loss = []

        # Split the dataset into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)         
        print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
        num_samples = len(x_train)

        for epoch in range(1, epochs + 1):

            # Randomly shuffle the training data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            print(f"---------- Epoch: {epoch} ----------")
        
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                # Forward Propagation
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                y_pred_train = self.forward_propagation(x_batch, activ_fn=activationfn)
                train_accuracy = self._calculate_accuracy(y_batch, y_pred_train)                         

                # Backward Propagation
                train_loss = self.backward_propagation(epoch, x_batch, y_batch, optimizer = optimizer, lr=lr, momentum=momentum,
                                                 beta=beta, beta1=beta1, beta2=beta2, epsilon=eps,
                                                 loss_fn = lossfn, activ_fn = activationfn, no_grad=False)
                
                # Predictions and accuracy on the Validation Set (x_val, y_val)
                y_pred_val = self.forward_propagation(x_val, activ_fn=activationfn)
                val_accuracy = self._calculate_accuracy(y_val, y_pred_val)
                val_loss = self.backward_propagation(epoch, x_val, y_val, optimizer = optimizer, lr=lr, momentum=momentum,
                                                 beta=beta, beta1=beta1, beta2=beta2, epsilon=eps,
                                                 loss_fn = lossfn, activ_fn = activationfn, no_grad=True)
                
                self.training_loss.append(train_loss)                
                self.validation_loss.append(val_loss)

            if epoch % 5 == 0:
                print(f"Training  : Epoch {epoch}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
                print(f"Validation: Epoch {epoch}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

            #wandb.log({'train loss': train_loss, 'train accuracy': train_accuracy * 100, 'epoch': epoch})
            #wandb.log({'val_accuracy': val_accuracy * 100})

    #########################################################################################################################
    '''CALCULATE ACCURACY'''
    
    def _calculate_accuracy(self, y_true, y_pred):
        correct = 0
        for i in range(len(y_true)):
            if np.argmax(y_true[i]) == np.argmax(y_pred[i]):
                correct += 1

        accuracy = correct / len(y_true)
        return accuracy
    
    #########################################################################################################################
    '''PREDICTIONS'''

    def predict(self, x_test, y_test, activationfn):        

        y_pred = self.forward_propagation(x_test, activ_fn=activationfn)
        accuracy = self._calculate_accuracy(y_test, y_pred)

        print(f"Test Accuracy: {accuracy:.4f}")

    #########################################################################################################################
    '''GET() METHODS TO EXTRACT VARIABLES FROM THE CLASS'''
    def get_parameters(self):
        return self.parameters    

    def get_training_loss(self):
        return self.training_loss

    def get_validation_loss(self):
        return self.validation_loss
    
    