import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn import preprocessing
import statistics
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


def how_many_na_col(data):
    """get the number of missing data per column"""
    missing_values_count = data.isnull().sum()
    missing_values_count.append(data.isnull().sum() * 100 / len(data))
    return missing_values_count[0:]



def csv_input_fn(csv_path):
    dataset = pd.read_csv(csv_path)
    dataset = dataset[1:]
    dataset = pd.DataFrame(dataset)
    dataset.replace({"Up": 1, "Left": 1, "Left-Up-Left": 1, "Up-Left": 1, "Up-Up-Left": 1,
                           "Down-Right": 0, "Down": 0, "Left-Down-Left": 0, "Down-Down-Right": 0, "Right-Up-Right": 0,
                           "Down-Left": 0, "Right": 0, "Right-Down-Right": 0, "Up-Right": 0, "Down-Down-Left": 0,
                           "Up-Up-Right": 0}, inplace=True)


    dataset.pop('Mouse_activity_1')
    dataset.pop('Mouse_activity_2')
    dataset.pop('Mouse_activity_3')
    dataset.pop('Unnamed: 0')
    dataset.pop('Date')
    dataset.pop('Location')
    dataset.pop('Time')
    dataset.pop('User_ID')


    dataset.pop('Dispatch_loc')
    dataset['Gender'].replace("F", 1, inplace=True)
    dataset['Gender'].replace("M", 0, inplace=True)
    dataset['Gender'].fillna(0, inplace=True)
    dataset['Bought_premium'].replace("Yes", 1, inplace=True)  # replace Yes to 1 # explain why
    dataset['Bought_premium'].replace("No", 0, inplace=True)  # replace No to 0 # explain why

    dataset.dropna(axis=0, thresh=1, subset=["Buy_premium"], inplace=True)  # throws out all the rows with
    # NA at the target variable
    dataset.dropna(axis=0, thresh=1, subset=["Bought_premium"], inplace=True)  # throws out all the rows with
    # NA at the target variable

    dataset['Min_prod_time'].fillna(dataset['Min_prod_time'].median(), inplace=True)
    dataset['Max_prod_time'].fillna(dataset['Max_prod_time'].median(), inplace=True)
    dataset['Commercial_1'].fillna(dataset['Commercial_1'].median(), inplace=True)
    dataset['Commercial_2'].fillna(dataset['Commercial_2'].median(), inplace=True)
    dataset['Commercial_3'].fillna(dataset['Commercial_3'].median(), inplace=True)
    dataset['Jewelry'].fillna(dataset['Jewelry'].median(), inplace=True)
    dataset['Shoes'].fillna(dataset['Shoes'].median(), inplace=True)
    dataset['Clothing'].fillna(dataset['Clothing'].median(), inplace=True)
    dataset['Home'].fillna(dataset['Home'].median(), inplace=True)
    dataset['Premium'].fillna(dataset['Premium'].median(), inplace=True)
    dataset['Idle'].fillna(dataset['Idle'].median(), inplace=True)
    dataset['Post_premium_commercial'].fillna(dataset['Post_premium_commercial'].median(), inplace=True)
    dataset['Premium_commercial_play'].fillna(dataset['Premium_commercial_play'].median(), inplace=True)
    dataset['Size_variations'].fillna(dataset['Size_variations'].median(), inplace=True)
    dataset['Color_variations'].fillna(dataset['Color_variations'].median(), inplace=True)

    dataset.replace('NA', 0, inplace=True)
    dataset.replace('nan', 0, inplace=True)
    dataset.replace('', 0, inplace=True)
    dataset.replace('None', 0, inplace=True)
    dataset.replace('NAN', 0, inplace=True)
    dataset.replace('NAT', 0, inplace=True)
    label = dataset.pop('Buy_premium')

    dataset = np.asarray(dataset).astype(np.float32)

    dataset = preprocessing.minmax_scale(dataset, feature_range=(0, 1), axis=0, copy=True)


    return dataset, label


if __name__ == '__main__':

    (x_train, y_train) = csv_input_fn("yos_dataset_test.csv")
    model = keras.Sequential([
        keras.layers.Dense(units=12, activation='relu'),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
              #loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              loss='mse',
              metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=10,
        steps_per_epoch=15,
        validation_steps=5
    )

    (x_validate, y_validate) = csv_input_fn("ctr_dataset_test.csv")
    y_pred = model.predict(x_validate)
#Converting predictions to label
    pred = list()
    median = statistics.median(y_pred)
    for i in range(len(y_pred)):
        if y_pred[i] > median:
            pred.append(1)
        else:
            pred.append(0)

    #print(head(pred,50))
    print(pred)

    print(y_validate)

    a = accuracy_score(pred,y_validate)
    print (a)