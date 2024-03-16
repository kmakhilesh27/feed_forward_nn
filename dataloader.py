from keras.datasets import fashion_mnist, mnist
from keras.utils import to_categorical

def load_and_process_dataset(choice = "fashion_mnist"):

    if choice == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif choice == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        raise Exception("Invalid Dataset.")

    # reshape the images into n-dimensional vector
    x_train = x_train.reshape(x_train.shape[0], -1)    
    x_test = x_test.reshape(x_test.shape[0], -1)

    # normalize x_train and x_test pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # one-hot encode y_train and y_test values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train , x_test, y_test