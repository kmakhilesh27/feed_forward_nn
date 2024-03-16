import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

wandb.init(project="CS6910_Assignment_1", entity="ge23m019")

def visualize_dataset(x_train, y_train):
    classes = {0:'T-Shirt/Top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat',
               5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle Boot'}   
    
    _, axs = plt.subplots(2, 5, figsize=(8, 20))
    images = []
    labels = []
    
    for i, ax in enumerate(axs.flatten()):
        idx = np.argmax(y_train == i)
        img = x_train[idx,:,:]        
        ax.imshow(img, cmap = 'gray')
        ax.set_title(classes[i])
        
        images.append(img)
        labels.append(classes[i])
    plt.show()
    wandb.log({"run_visualization": [wandb.Image(img, caption=caption) for img, caption in zip(images, labels)]})

if __name__=="__main__":
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    visualize_dataset(x_train, y_train)