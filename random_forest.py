# Final Project TAU neural networks.
# By Roi Yaacovi 206044844 & Yossi Dvori 021784665
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
from sklearn.metrics import roc_auc_score
import statistics
from sklearn.metrics import accuracy_score
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, SMOTENC
import shap
from sklearn import preprocessing
import time


# to do list:
# 12. use random forest information gain in order to determine which features are more important
# 19. explainable AI - local explain
print("start")


def plot_cor_matrix(data):
    """receives data and plot a heat map of the correlation matrix and a table of the unique features correlation
    returns the table of the unique features correlation"""
    correlation_matrix = data.corr()  # creating correlation_matrix
    # print(correlation_matrix)
    dataplot = sb.heatmap(correlation_matrix)  # creating a heat map of the correlation_matrix
    mp.show()  # showing the correlation_matrix
    sorted_mat = correlation_matrix.unstack().sort_values()
    # Retain upper triangular values of correlation matrix and
    # make Lower triangular values Null
    upper_corr_mat = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    # Convert to 1-D series and drop Null values
    unique_corr_pairs = upper_corr_mat.unstack().dropna()
    # Sort correlation pairs
    sorted_mat1 = unique_corr_pairs.sort_values()
    print(sorted_mat1)
    return sorted_mat1


# functions to handle Na values
def how_many_na_col(data):
    """get the number of missing data per column"""
    missing_values_count = data.isnull().sum()
    missing_values_count.append(data.isnull().sum() * 100 / len(data))
    return missing_values_count[0:]


def how_many_na_col_percentage(data):
    """get the percentage of missing data per column"""
    missing_values_count = (data.isnull().sum() * 100 / len(data))
    return missing_values_count[0:]


def load_dataset(filename):
    return pd.read_csv(filename)  # loading the data


def data_exploration (train_dataset):
    print(train_dataset.head(10))  # print the first 10 rows
    print("")
    print("Unique values in each col and count")
    categorical_columns = ["Gender","Location","Mouse_activity_1","Mouse_activity_2","Mouse_activity_3","Dispatch_loc","Bought_premium"]
    for column in categorical_columns:
        print(train_dataset[column].value_counts())  # prints the unique values in each col and counts them

    print("")
    print("Data summaries")
    for column in train_dataset:
        print(train_dataset[column].describe(include="all", datetime_is_numeric=True))  # prints the data summaries


    print("")
    print("median of numeric")
    print(train_dataset[
              ["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
               "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes",
               "Clothing",
               "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
               "Color_variations", "Dispatch_loc", "Bought_premium", "Buy_premium"]].median())  # shows median

def remove_unneeded_features(train_dataset):
    # droping features
    print("Plotting correlation matrix")
    train_dataset = train_dataset.drop(columns="User_ID")  # dropping the user_Id column.

    print("dropping the user_Id column.  has no meaning.")
    train_dataset = train_dataset.drop(columns="Unnamed: 0")  # dropping the Unnamed: 0 column.
    # it seems that it comes from the csv numbering

    print("dropping the Unnamed: 0 column. 0 correlation with all of the features, it seems that it comes from the csv numbering")
    train_dataset = train_dataset.drop(columns="Date")  # droping the date column  doesn't contribute to prediction
    # moreover we create based on him other columns
    print(" dropping the Date column. doesn't contribute moreover we create based on him other columns")
    train_dataset = train_dataset.drop(columns="day")  # droping the date column  doesn't contribute to prediction
    print(" dropping the day column. doesn't contribute to prediction")
    train_dataset = train_dataset.drop(columns="day of year")  # droping the day of year column  doesn't contribute to prediction
    print(" dropping the day of year column. doesn't contribute to prediction")
    train_dataset = train_dataset.drop(columns="Time")  # droping the Time column  doesn't contribute to prediction
    print(" dropping the Time column. doesn't contribute to prediction")
    train_dataset = train_dataset.drop(columns="Dispatch_loc")  # droping the Dispatch_loc column  doesn't contribute to prediction
    print(" dropping the Dispatch_loc column. doesn't contribute to prediction")
    #train_dataset = train_dataset.drop(columns="Mouse_activity_1")
    #train_dataset = train_dataset.drop(columns="Mouse_activity_2")
    #train_dataset = train_dataset.drop(columns="Mouse_activity_3")
    #train_dataset = train_dataset.drop(columns="Location")
    #train_dataset = train_dataset.drop(columns="month")
    #train_dataset = train_dataset.drop(columns="Gender")
    return train_dataset


def normalize_dataset(train_dataset):
    scaler = StandardScaler()
    # print(train_dataset.head(10))
    num_cols = train_dataset.columns[
        train_dataset.dtypes.apply(lambda c: np.issubdtype(c, np.number))]  # index of numric colums
    num_cols = num_cols.drop("Buy_premium")  # dropping Buy_premium from index
    num_cols = num_cols.drop("Unnamed: 0")  # dropping Unnamed: 0 from index
    num_cols = num_cols.drop("User_ID")  # dropping User_ID from index
    print(num_cols)
    scaler.fit(train_dataset[num_cols])
    print(scaler.mean_)
    train_dataset[num_cols] = scaler.transform(train_dataset[num_cols])
    return pd.DataFrame(train_dataset)


def na_handling(train_dataset):
    # between Male and Female in order to predict target variable

    train_dataset['Gender'].fillna(train_dataset['Gender'].mode()[0],
                                   inplace=True)  # there is no significant difference
    # between Male and Female in order to predict target variable
    print("")
    print("imputing Gender NA Values with mode Because because most of the entries are male.")
    train_dataset['Location'].fillna(value=0, inplace=True)
    print("")
    print("imputing Location NA Values with 0 Because we rather have FN than FP.")
    # train_dataset['Time'].fillna(train_dataset['Time'].mode()[0], inplace=True) # droped
    # train_dataset['Date'].fillna(train_dataset['Date'].mode()[0], inplace=True) # droped
    train_dataset['Mouse_activity_1'].fillna(value=0,
                                             inplace=True)  # imputing Mouse_activity NA values with 0 because we
    train_dataset['Mouse_activity_2'].fillna(value=0, inplace=True)  # rather have FN than FP.
    train_dataset['Mouse_activity_3'].fillna(value=0, inplace=True)
    print("")
    print("imputing Mouse_activity_1/2/3 NA Values with 0 Because we rather have FN than FP.")
    # train_dataset['Dispatch_loc'].fillna(train_dataset['Dispatch_loc'].mode()[0], inplace=True) # droped
    train_dataset['Bought_premium'].fillna(value=0, inplace=True)  # imputing Bought_premium NA Values with 0
    # Because we rather have FN than FP.
    print("")
    print("imputing Bought_premium NA Values with NO Because we rather have FN than FP.")

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

    missing_values_count = (train_dataset.isnull().sum() * 100 / len(train_dataset))
    print("")
    print("displaying Na count for each feature after NA handling")
    print(missing_values_count)
    return train_dataset


def features_engineering(train_dataset):
    # feature engineering for Date
    print("")
    print("creating new date related features")
    train_dataset['Date'].fillna(train_dataset['Date'].mode()[0], inplace=True)  # 0.068555 values are Na imputing here
    train_dataset['Date'] = pd.to_datetime(train_dataset['Date'], dayfirst=True)  # transform the date to type datetime
    train_dataset["day"] = train_dataset.apply(lambda row: row.Date.day_name(),
                                               axis=1)  # creates a new col with day name
    train_dataset["month"] = train_dataset.apply(lambda row: row.Date.month_name(),
                                                 axis=1)  # creates new col with month name
    train_dataset["day of year"] = train_dataset.apply(lambda row: str(row.Date.day_of_year),
                                                       axis=1)  # creates new col with month name

    # feature engineering for Mouse_activity_1/2/3
    train_dataset.replace({"Up": 1, "Left": 1, "Left-Up-Left": 1, "Up-Left": 1, "Up-Up-Left": 1,
                           "Down-Right": 0, "Down": 0, "Left-Down-Left": 0, "Down-Down-Right": 0, "Right-Up-Right":0,
                           "Down-Left": 0, "Right": 0, "Right-Down-Right": 0, "Up-Right": 0, "Down-Down-Left": 0,
                           "Up-Up-Right":0}, inplace=True)
    print("")
    print("transforming the Mouse_activity columns. Up, Left, Left-Up-Left, Up-Left, Up-Up-Left are replaced with 1"
          "the rest replaced with 0. this is because the 1 values have a higher chances of positive buy")
    print(train_dataset["Mouse_activity_1"].value_counts())
    print(train_dataset["Mouse_activity_2"].value_counts())
    print(train_dataset["Mouse_activity_3"].value_counts())


    train_dataset.replace({"Nof Hagalil": 1, "Dimona": 1, "Tamra": 1, "Haifa": 1, "Akko": 1, "Migdal HaEmek ": 1, "Safed": 1,
                           "Kiryat Gat": 1, "Migdal HaEmek": 1, "Hadera": 1, "Maalot Tarshiha": 1, "Harish": 1, "Kiryat Motzkin": 1,
                           "Rehovot": 1, "Herzliya": 1, "Ramla": 1, "Beer Sheva": 1, "Hod HaSharon": 1, "Tel Aviv": 1,
                           "Kiryat Ono": 1, "Tiberias": 1, "Yavne": 1, "Jerusalem": 1, "Beit Shemesh": 1, "Kfar Sava": 1,

                           "Afula": 0, "Raanana": 0,"Arad": 0,"Nes Ziona": 0,"Karmiel": 0,"Modiin": 0,"Nazareth": 0,
                           "Sakhnin": 0,"Ashkelon": 0, "Eilat": 0,"Beit Shean": 0,"Petah Tikva": 0,"Netanya": 0,"Shefaram": 0,
                           "Nahariya": 0,"Holon": 0,"Rishon Lezion": 0,"Kiryat Shemone": 0,
                           "Ramat Gan": 0,"Kiryat Bialik": 0,"Givatayim": 0,"Kiryat Ata": 0,"Ashdod": 0,"Yokneam": 0,
                           "Sderot": 0}, inplace=True)
    print("")
    print(train_dataset["Location"].value_counts())
    print("transforming the Location column. Nof Hagalil, Dimona, Tamra, Haifa, Akko, Migdal HaEmek, Safed,"
          "Kiryat Gat, Migdal HaEmek, Hadera, Maalot Tarshiha, Harish, Kiryat Motzkin,"
          "Rehovot, Herzliya, Ramla, Beer Sheva, Hod HaSharon, Tel Aviv,"
          "Kiryat Ono, Tiberias, Yavne, Jerusalem, Beit Shemesh, Kfar Sava replaced with 1"
          "the rest replaced with 0. this is because the 1 values have a higher chances of positive buy")

    # feature engineering for Bought_premium
    train_dataset.replace({"Yes": 1, "No": 0}, inplace=True)
    print("")
    print(train_dataset["Bought_premium"].value_counts())
    print("transforming the Bought_premium columns. Yes are replaced with 1"
          "NO with 0. this is because the 1 values have a higher chances of positive buy")

    train_dataset.replace({"F": 1, "M": 0}, inplace=True)
    print("")
    print(train_dataset["Gender"].value_counts())
    print("transforming the Gender columns. Female are replaced with 1"
          "Male with 0. for the model to work with")

    train_dataset.replace({"July": 1, "June": 1, "August": 1, "September": 0, "May": 0, "April": 0,
                           "November": 0, "March": 0, "December": 0, "October": 0, "January": 0, "February":0
                           }, inplace=True)
    print("")
    print(train_dataset["month"].value_counts())
    print("transforming the month columns. July, June and August are replaced with 1"
          "the rest replaced with 0. this is because the 1 values have a higher chances of positive buy")

    # feature engineering for Bought_premium
    train_dataset.replace({"Yes": 1, "No": 0}, inplace=True)
    print("")
    print(train_dataset["Bought_premium"].value_counts())
    print("transforming the Bought_premium columns. Yes are replaced with 1"
          "NO with 0. this is because the 1 values have a higher chances of positive buy")


    return train_dataset


def date_exploration(train_dataset):
    print("")
    print("day sorting (percentage of positive buy premium)")
    day_df = train_dataset.drop(
        columns=["User_ID", "Unnamed: 0", "Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time",
                 "Commercial_1", "Commercial_2",
                 "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes",
                 "Clothing",
                 "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
                 "Color_variations", "month", "day of year", "Dispatch_loc", "Bought_premium"])
    grouped = day_df.groupby("day")
    VCL = train_dataset["day"].value_counts()
    GSL = grouped.sum()
    GSL["How many"] = VCL
    GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]
    print(GSL.sort_values(by="percentage"))

    print("")
    print("month sorting (percentage of positive buy premium)")
    month_df = train_dataset.drop(
        columns=["User_ID", "Unnamed: 0", "Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time",
                 "Commercial_1", "Commercial_2",
                 "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes",
                 "Clothing",
                 "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
                 "Color_variations", "day", "day of year", "Dispatch_loc", "Bought_premium"])
    grouped = month_df.groupby("month")
    VCL = train_dataset["month"].value_counts()
    GSL = grouped.sum()
    GSL["How many"] = VCL
    GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]
    print(GSL.sort_values(by="percentage"))

    print("")
    print("day of year sorting (percentage of positive buy premium)")
    day_of_year_df = train_dataset.drop(
        columns=["User_ID", "Unnamed: 0", "Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time",
                 "Commercial_1", "Commercial_2",
                 "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes",
                 "Clothing",
                 "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
                 "Color_variations", "day", "month", "Dispatch_loc", "Bought_premium"])
    grouped = day_of_year_df.groupby("day of year")
    VCL = train_dataset["day of year"].value_counts()
    GSL = grouped.sum()
    GSL["How many"] = VCL
    GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]
    print(GSL.sort_values(by="percentage"))

def explore_cat_vals(train_dataset):
    #### categorical variables exploration ####
    print("")
    print("we now display for each unique value of categorical feature the number of positive Buy_premium")
    print("")
    print("Gender sorting percentage of positive buy premium)")
    gender_df = train_dataset.drop(
        columns=["User_ID", "Unnamed: 0", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1",
                 "Commercial_2",
                 "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes",
                 "Clothing",
                 "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
                 "Color_variations", "Dispatch_loc", "Bought_premium"])  # dropping all features except for
    # the one we want to create the table for
    grouped = gender_df.groupby("Gender")  # creating a grouped object that contains rows that are the target column
    VCL = train_dataset["Gender"].value_counts()  # counting and storing the unique values in the target column
    GSL = grouped.sum()  # summing the Buy_premium target variable for each unique value
    GSL["How many"] = VCL  # combing the VCL and GSL
    GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]  # # creating a percentage column
    print(GSL.sort_values(by="percentage"))  # printing and sorting

    print("")
    print("Location sorting percentage of positive buy premium)")
    Location_df = train_dataset.drop(
        columns=["User_ID", "Unnamed: 0", "Gender", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1",
                 "Commercial_2",
                 "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes",
                 "Clothing",
                 "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
                 "Color_variations", "Dispatch_loc", "Bought_premium"])
    grouped = Location_df.groupby("Location")
    VCL = train_dataset["Location"].value_counts()
    GSL = grouped.sum()
    GSL["How many"] = VCL
    GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]
    print(GSL.sort_values(by="percentage"))

    print("")
    print("Mouse_activity_1 sorting (percentage of positive buy premium)")
    Mouse_activity_1_df = train_dataset.drop(
        columns=["User_ID", "Unnamed: 0", "Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time",
                 "Commercial_1", "Commercial_2",
                 "Commercial_3", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
                 "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
                 "Color_variations", "Dispatch_loc",  "Bought_premium"])
    grouped = Mouse_activity_1_df.groupby("Mouse_activity_1")
    VCL = train_dataset["Mouse_activity_1"].value_counts()
    GSL = grouped.sum()
    GSL["How many"] = VCL
    GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]
    print(GSL.sort_values(by="percentage"))

    print("")
    print("Mouse_activity_2 sorting (percentage of positive buy premium)")
    Mouse_activity_2_df = train_dataset.drop(
        columns=["User_ID", "Unnamed: 0", "Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time",
                 "Commercial_1", "Commercial_2",
                 "Commercial_3", "Mouse_activity_1", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
                 "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
                 "Color_variations", "Dispatch_loc",  "Bought_premium"])
    grouped = Mouse_activity_2_df.groupby("Mouse_activity_2")
    VCL = train_dataset["Mouse_activity_2"].value_counts()
    GSL = grouped.sum()
    GSL["How many"] = VCL
    GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]
    print(GSL.sort_values(by="percentage"))

    print("")
    print("Mouse_activity_3 sorting (percentage of positive buy premium)")
    Mouse_activity_3_df = train_dataset.drop(
        columns=["User_ID", "Unnamed: 0", "Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time",
                 "Commercial_1", "Commercial_2",
                 "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Jewelry", "Shoes", "Clothing",
                 "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
                 "Color_variations", "Dispatch_loc",  "Bought_premium"])
    grouped = Mouse_activity_3_df.groupby("Mouse_activity_3")
    VCL = train_dataset["Mouse_activity_3"].value_counts()
    GSL = grouped.sum()
    GSL["How many"] = VCL
    GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]
    print(GSL.sort_values(by="percentage"))

    print("")
    print("Dispatch_loc sorting (percentage of positive buy premium)")
    Dispatch_loc_df = train_dataset.drop(
        columns=["User_ID", "Unnamed: 0", "Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time",
                 "Commercial_1", "Commercial_2",
                 "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes",
                 "Clothing",
                 "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
                 "Color_variations", "Bought_premium"])
    grouped = Dispatch_loc_df.groupby("Dispatch_loc")
    VCL = train_dataset["Dispatch_loc"].value_counts()
    GSL = grouped.sum()
    GSL["How many"] = VCL
    GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]
    print(GSL.sort_values(by="percentage"))

    print("")
    print("Bought_premium sorting (percentage of positive buy premium)")
    Bought_premium_df = train_dataset.drop(
        columns=["User_ID", "Unnamed: 0", "Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time",
                 "Commercial_1", "Commercial_2",
                 "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes",
                 "Clothing",
                 "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
                 "Color_variations", "Dispatch_loc"])
    grouped = Bought_premium_df.groupby("Bought_premium")
    VCL = train_dataset["Bought_premium"].value_counts()
    GSL = grouped.sum()
    GSL["How many"] = VCL
    GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]
    print(GSL.sort_values(by="percentage"))

########################################## MAIN ###########################################

train_dataset = load_dataset("ctr_dataset_train.csv")
# Data Exploration
data_exploration(train_dataset)

# print("")
# print("Data normalization - we chose not to normalize the data because we use Random forest,"
#       "and the model doesn't assume normalized data")

#Data normalization - we chose to normalize the data because we use Neural Network,
# and the model assume normalized data
print("")
print("Data normalization - we chose to normalize the data because we use Neural Network,"
      "and the model assume normalized data. Neural Network is built from logistic regression models"
      "and logistic regression needs to get normalized data"
      "notice that because we normalize the data the performance of the random forest are decreasing"
      "if we won't normalized the data than random forest will be better but still the Neural Network"
      "surpassed the random forest")
train_dataset = normalize_dataset(train_dataset)

# Balanced data
print("")
print("We chose not to balanced the data because balancing damaged the model prediction")
print("We chose SMOTE -Synthetic Minority Oversampling TEchnique. SMOTE creates new samples of the minority Data")
print("in our case the positive but data. it creates it by looking at the minority data and creating similar rows")
print("this can help the model by balancing the data. when model are created with unbalanced data they tend to be")
print("not as good in prediction on the minority data")


#Explore categorical vars:
explore_cat_vals(train_dataset)

#plotting correlation matrix:
plot_cor_matrix(train_dataset)  # creating cor mat for only the numeric

###################### feature engineering ######################
train_dataset = features_engineering(train_dataset)

#Explore the new date var
date_exploration(train_dataset)

##################### remove features ###########################
train_dataset = remove_unneeded_features(train_dataset)


# Plotting correlation matrix after dropped features
print("")
print("Plotting correlation matrix after dropped some features only the numeric and undropped variables"
      "variables that are droped are not relevent anymore")
plot_cor_matrix(train_dataset)  # creating cor mat for only the numeric and undropped variables.





# Fixing negative values - we assume that the negative values represent people who didn't Post_premium_commercial
train_dataset[train_dataset['Idle'] < 0] = 0
train_dataset[train_dataset['Post_premium_commercial'] < 0] = 0
# print_data_summaries(train_dataset)




####### NA handling #######
print("")
print("number of lines in the data")
print(len(train_dataset.index))  # number of lines in the data
print("")
print("number of na in each col")
print(how_many_na_col(train_dataset))  # number of na in each col
print("")
print("percentage of na in each col")
print(how_many_na_col_percentage(train_dataset))  # percentage of na in each col

# עמודה עם יותר מ-60% ערכים חסרים היא מיותרת
# Color_variations העמודות Commercial_2 ו Commercial_3 ו Size_variations
# יש להן יותר מ 40% ערכים נעלמים אבל בחרתי לא להוריד אותן בנתיים


# dropping col or rows
# new_train_dataset = train_dataset.dropna(axis=0, thresh=10)  # axis=0 means rows (1 means columns)
# thresh=10 means that if we have at least 10 non Na values we keep the row can also use subset=["colname1",
# "colname2"] - tells where to check for an valuse - after correlation and information gain we can decide
# feature_engineering(train_dataset)

# throws out all the rows with NA at the target variable
# we chose to throws out all the rows with NA at the target variable because we can't train the model on htem
# we could use KNN to try to predict the target variable but that is risky becuase it a prediction on predication
# moreover the number we dispoed is not that big and shouln't affect the results
print("")
print(len(train_dataset.index))
print("this is the number of rows in the data frame")
new_train_dataset = train_dataset.dropna(axis=0, thresh=1, subset=[
    "Buy_premium"])  # throws out all the rows with NA at the target variable
print(len(new_train_dataset.index))
print("this is the number of rows after we throws out all the rows with NA at the target variable")
print("we dispoed of"+str(len(train_dataset.index) - len(new_train_dataset.index))+"rows")
train_dataset.dropna(axis=0, thresh=1, subset=["Buy_premium"], inplace=True)  # throws out all the rows with
# NA at the target variable
print(len(train_dataset.index))
print("this is the number of rows in the data frame after throw")
print("# we chose to throws out all the rows with NA at the target variable because we can't train the model on them")
print("we could use KNN to try to predict the target variable but that is risky becuase it a prediction on predication")
print("moreover the number we dispoed is not that big and shouln't affect the results")


# impute categorical data
missing_values_count = (train_dataset.isnull().sum() * 100 / len(train_dataset))
print("")
print("displaying Na count for each feature")
print(missing_values_count)


#handle na
train_dataset = na_handling(train_dataset)

# Data splitting
# Let's say we want to split the data in 70:15:15 for train:valid:test dataset
print("")
print("data splitting: 70% train, 15% validation 15% internal test")
train_size = 0.7


feature_vector = train_dataset.drop(columns=['Buy_premium']).copy()
target_variable = train_dataset['Buy_premium']

# In the first step we will split the data in training and remaining dataset to be split later to validation and test
feature_vector_train, feature_vector_to_split, target_variable_train, target_variable_to_split = train_test_split(
    feature_vector, target_variable, train_size=0.7, random_state=0)
# random_state=0 is important because it will give us the same split data everytime

# Now since we want the valid and test size to be equal (15% each of overall data).
# we have to define test_size=0.5 (that is 50% of remaining data)
feature_vector_valid, feature_vector_test, target_variable_valid, target_variable_test = train_test_split(
    feature_vector_to_split, target_variable_to_split, test_size=0.5, random_state=0)

print(feature_vector_train.shape), print(target_variable_train.shape)
print(feature_vector_valid.shape), print(target_variable_valid.shape)
print(feature_vector_test.shape), print(target_variable_test.shape)

# random forest information gain feature selection
# should probably do also before imputing the data and check if there is any different results and also after
# a lot of things can affect the importance


############################################################# Neural Network ######################################
print("Neural Network Creation:")
model = keras.Sequential([
        keras.layers.Dense(units=12, activation='relu'),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

print("Neural Network Training:")
history = model.fit(
        feature_vector_train, target_variable_train,
        epochs=20,
        steps_per_epoch=50,
        validation_steps=15
    )

print("Neural Network Prediction on validation data:")
y_pred = model.predict(feature_vector_valid)

#Converting predictions to label
pred = list()
median = statistics.median(y_pred)
for i in range(len(y_pred)):
    if y_pred[i] > (median + 0.05):
       pred.append(1)
    else:
       pred.append(0)

a = accuracy_score(pred, target_variable_valid)
print("Neural Network Accuracy for Validation data is: " + str(a))

confusion_matrix_neural_network_valid = confusion_matrix(target_variable_valid, pred,
                                    labels=[1, 0])  # create confusion_matrix
print('Neural Network Confusion matrix for validation data:\n\n', confusion_matrix_neural_network_valid)
print("TP, FN")
print("FP, TN")

display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_neural_network_valid)  # create an onbject to display the confusion_matrix
display.plot()

print(classification_report(target_variable_valid,
                            pred))  # create classification_report of varius indicators

print("Neural Network with Validation data AUC score:" + str(roc_auc_score(target_variable_valid, pred)))

print("Neural Network predict the test data")
y_pred = model.predict(feature_vector_test)
#print (y_pred)
#Converting predictions to label
pred = list()
median = statistics.median(y_pred)
for i in range(len(y_pred)):
    if y_pred[i] > (median+0.05):
       pred.append(1)
    else:
       pred.append(0)

a = accuracy_score(pred, target_variable_test)
print("Neural Network Accuracy for Test data is: " + str(a))

confusion_matrix_neural_network_test = confusion_matrix(target_variable_test, pred,
                                    labels=[1, 0])  # create confusion_matrix
print('Neural network Confusion matrix for test data:\n\n', confusion_matrix_neural_network_test)
print("TP, FN")
print("FP, TN")

display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_neural_network_test)  # create an onbject to display the confusion_matrix
display.plot()

print(classification_report(target_variable_test,
                            pred))  # create classification_report of varius indicators

print("Neural Network AUC score for test data:" + str(roc_auc_score(target_variable_test, pred)))

############################################################ Random Forest #########################################


print("")
print("Random forest with 100 trees:")
# Model: Random forest with 100 trees # better results then 10 trees
cols = feature_vector_train.columns
scaler = RobustScaler()
feature_vector_train = scaler.fit_transform(feature_vector_train)
feature_vector_valid = scaler.transform(feature_vector_valid)

feature_vector_train = pd.DataFrame(feature_vector_train, columns=[cols])
feature_vector_valid = pd.DataFrame(feature_vector_valid, columns=[cols])

rfc = RandomForestClassifier(n_estimators=100, random_state=0)  # instantiate the classifier
rfc.fit(feature_vector_train, target_variable_train)  # fit the model
target_variable_prediction_on_train_validation = rfc.predict(feature_vector_valid)  # Predict the Test set results
print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(
    accuracy_score(target_variable_valid, target_variable_prediction_on_train_validation)))


confusion_matrix_random_forest = confusion_matrix(target_variable_valid, target_variable_prediction_on_train_validation,
                                    labels=[1, 0])  # create confusion_matrix
print('Confusion matrix\n\n', confusion_matrix_random_forest)
print("TP, FN")
print("FP, TN")

display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_random_forest)  # create an onbject to display the confusion_matrix
display.plot()

print(classification_report(target_variable_valid,
                            target_variable_prediction_on_train_validation))  # create classification_report of varius indicators

print("AUC score:" + str(roc_auc_score(target_variable_valid, target_variable_prediction_on_train_validation)))


target_variable_prediction_on_train_test = rfc.predict(feature_vector_test)  # Predict the Test set results
print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(
    accuracy_score(target_variable_test, target_variable_prediction_on_train_test)))

confusion_matrix_random_forest_test = confusion_matrix(target_variable_test, target_variable_prediction_on_train_test,
                                    labels=[1, 0])  # create confusion_matrix
print('Confusion matrix\n\n', confusion_matrix_random_forest_test)
print("TP, FN")
print("FP, TN")

display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_random_forest_test)  # create an onbject to display the confusion_matrix
display.plot()

print(classification_report(target_variable_test,
                            target_variable_prediction_on_train_test))  # create classification_report of varius indicators

print("AUC score:" + str(roc_auc_score(target_variable_test, target_variable_prediction_on_train_test)))

# importance
# feature_names = [f'feature {i}' for i in range(feature_vector_train.shape[1])]
# start_time = time.time()
# importances = rfc.feature_importances_
# std = np.std([
#     tree.feature_importances_ for tree in rfc.estimators_], axis=0)
# elapsed_time = time.time() - start_time
#
# print(f"Elapsed time to compute the importances: "
#       f"{elapsed_time:.3f} seconds")
# forest_importances = pd.Series(importances, index=feature_names)
#
# fig, ax = mp.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()



print("We did hyperparameter tuning on the number of trees")
print("we tried 10 trees, 100 trees and 200 trees")
print("the best results were with 100 trees")
print("the number of trees is the number of individual decision trees the random forest creates")
print("with low numbers the more we add the better the results will be")
print("but at a certain point the results will start to be worse due too overfitting")
print("moreover we tried to change the number of ")

# SHAP Value
print("")
# explainer = shap.TreeExplainer(rfc)  # Create object that can calculate shap values
# explainer = shap.KernelExplainer(model)  # Create object that can calculate shap values
print("explainer check")
# shap_values = explainer.shap_values(feature_vector_valid)  # Calculate Shap values
print("shap_values check")
print("The SHAP TreeExplainer was too heavy for us too run so"
      "we explain below as instructed in the model")
# shap.summary_plot(shap_values[1], feature_vector_valid)
print("")
print("A SHAP value interpret the impact of having a certain value for a given feature in comparison to the prediction "
      "we'd make if that feature took some baseline value")
print("The summary plot combines feature importance with feature effects. Each point on the summary plot"
      " is a Shapley value for a feature and an instance. The position on the y-axis is determined by the"
      " feature and on the x-axis by the Shapley value. The color represents the value of the feature from "
      "low to high. Overlapping points are jittered in y-axis direction, so we get a sense of the distribution "
      "of the Shapley values per feature. The features are ordered according to their importance. In  other words:"
      "the the y axis shows the feature value high value will be pink low value will be blue."
      "the X axis shows the SHAP VALUE of the feature for each predication. the features are also ordered "
      "in accordance with their importance. Overall we get a sense of how the feature effect on the predication,"
      "in terms of importance overall, how high or low value have impact on the shap value."
      "the higher the shap value the higher the importance")
print("summary_plot executed")
print("because we cant run the SHAP (it is too heavy) we cant also shap_plot on the raffled data"
      "but if we could than we could see how each feature contributed too the specific prediction"
      "we could see if the value of the feature was low or high and to which direction of the prediction"
      "it pulled and how strong pulled the prediction to that direction")


