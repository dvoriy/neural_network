import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np

https://www.bmc.com/blogs/create-neural-network-with-tensorflow/
https://mlfromscratch.com/neural-network-tutorial/  # /

def read_data_files_train(name):
    #test_dataset = pd.read_csv("ctr_dataset_test.csv")
    train_dataset = pd.read_csv("ctr_dataset_train.csv")
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 20)
    print(train_dataset.head(10))
    print('Data files loaded successfully!')
    return train_dataset

if __name__ == '__main__':
    train_dataset = pd.read_csv("ctr_dataset_train.csv")
    print (np.shape(train_dataset))
    model = tf.keras.Sequential([
        input_shape=(28),
        Dense(128, activation='sigmoid'),
        Dense(2)
    ])

    model.compile(optimizer='SGD',
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_dataset, train_dataset, epochs=10)