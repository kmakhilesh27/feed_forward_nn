import wandb
from dataloader import load_and_process_dataset
from argparser import create_parser
import neural_network as nn

sweep_config = {
        'method': 'grid',
        'name' : 'complete-hyperparameter-sweep',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
            },
        'parameters': {
            'epochs': {'values': [5,10]},
            'num_layers': {'values': [3,4,5]},
            'hidden_size':{'values':[32,64,128]},
            'learning_rate': {'values': [1e-3, 1e-4]},
            'optimizer': {'values': ['sgd','momentum','nesterov','rmsprop','adam','nadam']},
            'batch_size': {'values': [16,32,64]},
            'weight_init': {'values':['random','xavier']},                
            'activation': {'values': ['sigmoid','tanh','relu']},
            'loss': {'values': ['cross_entropy']}
            }
        }

parser = create_parser()
args = vars(parser.parse_args())

def wandb_runner(config = sweep_config, usr_args = args):
    with wandb.init(config = config):
        config = wandb.init().config
        wandb.run.name = "e_{}_nl_{}_hs_{}_lr_{}_opt_{}_bs_{}_wi_{}_ac_{}".format(
            config.epochs,
            config.num_layers,
            config.hidden_size,
            config.learning_rate,            
            config.optimizer,
            config.batch_size,
            config.weight_init,
            config.activation
            )
        model = nn.FeedForwardNN(input_size=784, num_hidden_layers=config.num_layers,
                             hidden_size=config.hidden_size, output_size=10,
                             weight_initializer=config.weight_init)
        model.train(x_train, y_train,
                    epochs = config.epochs,
                    batch_size = config.batch_size,
                    lossfn = config.loss,
                    optimizer = config.optimizer,
                    lr = config.learning_rate,
                    momentum = usr_args['momentum'],
                    beta = usr_args['beta'],
                    beta1 = usr_args['beta1'],
                    beta2 = usr_args['beta2'],
                    eps = usr_args['epsilon'],
                    weight_decay = usr_args['weight_decay'],
                    activationfn = usr_args['activation']
                    )
        model.predict(x_test, y_test, config.activation)

x_train, y_train , x_test, y_test = load_and_process_dataset(choice = args['dataset'])

sweep_id = wandb.sweep(sweep=sweep_config, project='CS6910_Assignment_1')   
wandb.agent(sweep_id, function=wandb_runner, count = 10)
wandb.finish()