from dataloader import load_and_process_dataset
from argparser import create_parser
import neural_network as nn

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
import wandb, io

parser = create_parser()
args = vars(parser.parse_args())

"""
args = {'wandb_project': 'CS6910_Assignment_1', 'wandb_entity': 'ge23m019', 'dataset': 'fashion_mnist',
'epochs': 1, 'batch_size': 4, 'loss': 'cross_entropy', 'optimizer': 'sgd', 'learning_rate': 0.1,
'momentum': 0.5, 'beta': 0.5, 'beta1': 0.5, 'beta2': 0.5, 'epsilon': 1e-06, 'weight_decay': 0.0,
'weight_init': 'random', 'num_layers': 1, 'hidden_size': 4, 'activation': 'sigmoid'}
"""
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
                    activationfn = args['activation'],
                    use_wandb = False
                    )
y_test_pred = model.predict(x_test, y_test,args['activation'])

y_test_pred_decoded = []
y_test_decoded = []
for i in range(len(y_test_pred)):
    y_test_pred_decoded.append(np.argmax(y_test_pred[i]))
    y_test_decoded.append(np.argmax(y_test[i]))


def plot_confusion_matrix(y_true, y_pred, class_names, title=None, cmap=plt.cm.Blues):

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)    
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center"
                     )

    plt.tight_layout()
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()    

class_names = ['T-Shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

plot_confusion_matrix(y_test_decoded, y_test_pred_decoded, class_names)


wandb.init(project="CS6910_Assignment_1", entity="ge23m019")

wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    probs=None,
    y_true=y_test_decoded,
    preds=y_test_pred_decoded,
    class_names=class_names)})


