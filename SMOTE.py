# Final Project TAU neural networks.
# By Roi & Yossi D
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, SMOTENC
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
train_dataset = pd.read_csv("ctr_dataset_train.csv") # loading the data

# SMOTE is used to deal with unbalanced data set


# NA filling
train_dataset.dropna(axis=0, thresh=1, subset=["Buy_premium"], inplace=True) # throws out all the rows with NA in Buy_premium
train_dataset['Gender'].fillna(train_dataset['Gender'].mode()[0], inplace=True)
train_dataset['Location'].fillna(train_dataset['Location'].mode()[0], inplace=True)
train_dataset['Mouse_activity_1'].fillna(train_dataset['Mouse_activity_1'].mode()[0], inplace=True)
train_dataset['Time'].fillna(train_dataset['Time'].mode()[0], inplace=True)
train_dataset['Date'].fillna(train_dataset['Date'].mode()[0], inplace=True)
train_dataset['Mouse_activity_2'].fillna(train_dataset['Mouse_activity_2'].mode()[0], inplace=True)
train_dataset['Mouse_activity_3'].fillna(train_dataset['Mouse_activity_3'].mode()[0], inplace=True)
train_dataset['Dispatch_loc'].fillna(train_dataset['Dispatch_loc'].mode()[0], inplace=True)
train_dataset['Bought_premium'].fillna(train_dataset['Bought_premium'].mode()[0], inplace=True)

# impute numeric data with median
train_dataset['Min_prod_time'].fillna(train_dataset['Min_prod_time'].median(), inplace=True)
train_dataset['Max_prod_time'].fillna(train_dataset['Max_prod_time'].median(), inplace=True)
train_dataset['Commercial_1'].fillna(train_dataset['Commercial_1'].median(), inplace=True)
train_dataset['Commercial_2'].fillna(train_dataset['Commercial_2'].median(), inplace=True)
train_dataset['Commercial_3'].fillna(train_dataset['Commercial_3'].median(), inplace=True)
train_dataset['Jewelry'].fillna(train_dataset['Jewelry'].median(), inplace=True)
train_dataset['Shoes'].fillna(train_dataset['Shoes'].median(), inplace=True)
train_dataset['Clothing'].fillna(train_dataset['Clothing'].median(), inplace=True)
train_dataset['Home'].fillna(train_dataset['Home'].median(), inplace=True)
train_dataset['Premium'].fillna(train_dataset['Premium'].median(), inplace=True)
train_dataset['Idle'].fillna(train_dataset['Idle'].median(), inplace=True)
train_dataset['Post_premium_commercial'].fillna(train_dataset['Post_premium_commercial'].median(), inplace=True)
train_dataset['Premium_commercial_play'].fillna(train_dataset['Premium_commercial_play'].median(), inplace=True)
train_dataset['Size_variations'].fillna(train_dataset['Size_variations'].median(), inplace=True)
train_dataset['Color_variations'].fillna(train_dataset['Color_variations'].median(), inplace=True)
# print(train_dataset.isnull().sum())
# print(train_dataset.dtypes)

# in order for SMOTEC to work need to convert to int32
train_dataset = train_dataset.astype({'Min_prod_time': 'int32', "Max_prod_time": 'int32', "Commercial_1": 'int32'
                      , "Commercial_2": 'int32', "Commercial_3": 'int32', "Jewelry": 'int32'
                      , "Shoes": 'int32', "Clothing": 'int32', "Home": 'int32'
                      , "Premium": 'int32', "Idle": 'int32', "Post_premium_commercial": 'int32'
                      , "Premium_commercial_play": 'int32', "Size_variations": 'int32', "Color_variations": 'int32'})
# print(train_dataset.dtypes)

feature_vector = train_dataset.drop(columns=['Buy_premium', "Unnamed: 0", "User_ID"]).copy() # creating feature_vector
target_variable = train_dataset['Buy_premium'] # creating target_variable
print(feature_vector.isnull().sum()) # NA check
print(feature_vector.dtypes) # type check
# data_frame =pd.DataFrame(
#     {
#         "A": [10, 20, 30, 30, 40, 50, 60, 70],
#         "B": [1, 1, 0, 1, 0, 0, 1, 1],
#         "C": [32, 523, 23, 42, 1, 31, 188, 23],
#         "D": ["x", "x", "z", "a", "v", "d", "d", "a"]
#     })
num_of_entries = 32000
# feature_vector = data_frame.drop(columns = ['B']).copy()
# target_variable = data_frame['B']
print(feature_vector.head(10))
smote = SMOTENC(random_state=100, categorical_features=[0,1,2,3,9,10,11,22,23]) # creating SMOTEC obkect
feature_vector_res_head, target_variable_res_head = smote.fit_resample(feature_vector.head(num_of_entries), target_variable.head(num_of_entries)) # transform the data
#print(feature_vector_res_head)
feature_vector_res_tail, target_variable_res_tail = smote.fit_resample(feature_vector.tail(num_of_entries), target_variable.tail(num_of_entries)) # transform the data

feature_vector_res = pd.concat([feature_vector_res_head, feature_vector_res_tail])
target_variable_res = pd.concat([target_variable_res_head, target_variable_res_tail])
print (target_variable_res.head())