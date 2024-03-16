from dataloader import load_and_process_dataset
from argparser import create_parser
import neural_network as nn

import matplotlib.pyplot as plt
import numpy as np

parser = create_parser()
args = vars(parser.parse_args())

x_train, y_train , x_test, y_test = load_and_process_dataset(choice = args['dataset'])

model = nn.FeedForwardNN(input_size=784, num_hidden_layers=args['num_layers'],
                             hidden_size=args['hidden_size'], output_size=10,
                             weight_initializer=args['weight_init'])
model.train(x_train, y_train,
            epochs = args['epochs'],
                    batch_size = args['batch_size'],
                    lossfn = args['loss'],
                    optimizer = args['optimizer'],
                    lr = args['learning_rate'],
                    momentum = args['momentum'],
                    beta = args['beta'],
                    beta1 = args['beta1'],
                    beta2 = args['beta2'],
                    eps = args['epsilon'],
                    weight_decay = args['weight_decay'],
                    activationfn = args['activation']
                    )
model.predict(x_test, y_test,args['activation'])

parameters = model.get_parameters()
train_loss = model.get_training_loss()
val_loss = model.get_validation_loss()
train_acc = model.get_training_accuracy()
val_acc = model.get_validation_accuracy()

# Plotting the training and validation loss
plt.figure()
num_epochs = len(train_loss)
plt.plot(np.arange(num_epochs), train_loss, color='blue', label='Training Loss')
plt.plot(np.arange(num_epochs), val_loss, color='red', label='Validation Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
num_epochs = len(train_acc)
plt.plot(np.arange(num_epochs), train_acc, color='blue', label='Training Accuracy')
plt.plot(np.arange(num_epochs), val_acc, color='red', label='Validation Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()