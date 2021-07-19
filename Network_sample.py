import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from neural_project import feature_vector_train
from neural_project import target_variable_train
'''
https://www.bmc.com/blogs/create-neural-network-with-tensorflow/
https://mlfromscratch.com/neural-network-tutorial/  # /
'''

def read_data_files_train(name):
    #test_dataset = pd.read_csv("ctr_dataset_test.csv")
    train_dataset = pd.read_csv("ctr_dataset_train.csv")
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 20)
    print(train_dataset.head(10))
    print('Data files loaded successfully!')
    return train_dataset

if __name__ == '__main__':

    # impute numeric data with median
    Min_prod_time = tf.feature_column.numeric_column ("Min_prod_time")
    Max_prod_time = tf.feature_column.numeric_column("Max_prod_time")
    Commercial_1 = tf.feature_column.numeric_column("Commercial_1")
    Commercial_2 = tf.feature_column.numeric_column("Commercial_2")
    Commercial_3 = tf.feature_column.numeric_column("Commercial_3")
    Jewelry = tf.feature_column.numeric_column("Jewelry")
    Shoes = tf.feature_column.numeric_column("Shoes")
    Clothing = tf.feature_column.numeric_column("Clothing")
    Idle = tf.feature_column.numeric_column("Idle")
    Post_premium_commercial = tf.feature_column.numeric_column("Post_premium_commercial")
    Premium_commercial_play = tf.feature_column.numeric_column("Premium_commercial_play")
    Size_variations = tf.feature_column.numeric_column("Size_variations")
    Color_variations = tf.feature_column.numeric_column("Color_variations")
    feature_columns = [Min_prod_time, Max_prod_time, Commercial_1, Commercial_2, Commercial_3, Jewelry, Shoes, Clothing, Idle, Post_premium_commercial, Premium_commercial_play,
                       Size_variations, Color_variations]

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=6,
        model_dir="/tmp")
    batch_size = 100
''' model = tf.keras.Sequential([
        input_shape=(28),
        Dense(128, activation='sigmoid'),
        Dense(2)
    ])

    model.compile(optimizer='SGD',
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_dataset, train_dataset, epochs=10)
'''