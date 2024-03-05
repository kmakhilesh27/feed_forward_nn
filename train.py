import numpy as np
import matplotlib.pyplot as plt

def visualize_data(x_train, y_train):
    dim = x_train.shape[1]
    n_class = np.unique(y_train)

    _, axs = plt.subplots(len(n_class), 10, figsize=(12, 12))
    
    for i, d in enumerate(n_class):
        for j in range(10):
            axs[i, j].imshow(x_train[y_train == d][j].reshape((dim, dim)), cmap='gray', interpolation='none')
            axs[i, j].axis('off')

    plt.show() 

def load_dataset(choice = "fashion_mnist"):
    from keras.datasets import fashion_mnist, mnist
    if choice == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif choice == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        raise Exception("Invalid dataset.")



