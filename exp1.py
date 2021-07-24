import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np

train_dataset = pd.read_csv("ctr_dataset_train.csv") # loading the data
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

FIELD_DEFAULTS = [[0], [0], [0], [0], [0],
                  [0], [0], [0], [0], [0],
                  [0], [0], [0], [0], [0],
                  [0], [0], [0], [0], [0],
                  [0], [0], [0], [0], [0],
                  [0], [0]]

feature_names = ["Gender", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2", "Commercial_3",
                "Jewelry", "Shoes", "Clothing", "Home", "Premium","Premium_commercial_play", "Idle", "Post_premium_commercial",
                  "Size_variations","Color_variations", "Bought_premium"]
def parse_line(line):
    if line.shape == ():
        return
    parsed_line = tf.io.decode_csv(line, FIELD_DEFAULTS)
    label = parsed_line[-1]
    del parsed_line[-1]
    del parsed_line[0]
    del parsed_line[1]
    parsed_line[2].replace("F", 1, inplace=True)  # replace F to 1 # explain why
    parsed_line[2].replace("M", 0, inplace=True)  # replace M to 0 # explain why
    del parsed_line[3]
    del parsed_line[4]
    del parsed_line[5]
    del parsed_line[11]
    del parsed_line[12]
    del parsed_line[13]
    del parsed_line[24]
    parsed_line[25].replace("Yes", 1, inplace=True)  # replace Yes to 1 # explain why
    parsed_line[25].replace("No", 0, inplace=True)  # replace No to 0 # explain why
    features = parsed_line
    d = dict(zip(feature_names, features))
    print("dictionary", d, " label = ", label)
    return d, label

def csv_input_fn(csv_path):
    dataset = pd.read_csv(csv_path)
    dataset = dataset[1:]
    dataset = pd.DataFrame(dataset)
    dataset.replace("NA", 0, inplace=True)
    dataset.pop('Unnamed: 0')
    dataset.pop('Date')
    dataset.pop('Location')
    dataset.pop('Time')
    dataset.pop('Mouse_activity_1')
    dataset.pop('Mouse_activity_2')
    dataset.pop('Mouse_activity_3')
    dataset.pop('Dispatch_loc')
    dataset['Gender'].replace("F", 1, inplace=True)
    dataset['Gender'].replace("M", 0, inplace=True)
    dataset['Bought_premium'].replace("Yes", 1, inplace=True)  # replace Yes to 1 # explain why
    dataset['Bought_premium'].replace("No", 0, inplace=True)  # replace No to 0 # explain why

    label = dataset.pop('Buy_premium')
    dataset = np.asarray(dataset).astype(np.float32)

    return dataset, label


if __name__ == '__main__':

    (x_train, y_train) = csv_input_fn("ctr_dataset_train.csv")
    model = keras.Sequential([
        keras.layers.Reshape(target_shape=(1 * 18,), input_shape=(1, 18)),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=192, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=1, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=10,
        steps_per_epoch=500,
        validation_steps=2
    )
    #classifier = tf.estimator.DNNClassifier(
     #   feature_columns=feature_columns,
     #   hidden_units=[10, 10],
     #   n_classes=3,
     #   model_dir="/tmp")
    #batch_size = 100

    #classifier.train(
    #    steps=1000,
    #    input_fn=lambda: csv_input_fn("ctr_dataset_train.csv", batch_size))
