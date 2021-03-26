import tensorflow as tf
import numpy as np
import os


def main():
    cwd = os.getcwd()
    load_data_and_save(cwd+"/data_sets/mnist/")
    print("Done loading...")


def load_data_and_save(path):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    with open(path+'train_images.npy', 'wb') as f:
        np.save(f, train_images)
    with open(path + 'train_labels.npy', 'wb') as f:
        np.save(f, train_labels)
    with open(path + 'test_images.npy', 'wb') as f:
        np.save(f, test_images)
    with open(path + 'test_labels.npy', 'wb') as f:
        np.save(f, test_labels)




if __name__ == "__main__":
    main()
