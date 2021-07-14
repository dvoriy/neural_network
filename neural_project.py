# Final Project TAU neural networks.
# By Roi & Yossi D
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
import numpy as np


# import pandas_profiling as pp # a libery that helps explore the data
# profile_report = pp.ProfileReport(train_dataset, title="profile report of train data set", minimal=True)
# profile_report.to_file("output.html")

# to do list:
# 1. split the data: train 70%; validation 15%; test 10%
# 2. determine which col if any have to many NA values and therefore are unnecessary
#    meaning that that we leave a feature outside
#    create a function
# 3. determine which rows if any have to many NA values and therefore are unnecessary
#    meaning that that we leave an client outside
#    create a function
# 4. determine how to handle NA values and write a function
# 5. clean the data - look for negative values etc and decided what to do
# 6. make sure that i have at least 5 summery statistics for each feature
# 7. maybe create a function for the location feature. map the locations, organize each town to an area: north, south
#    center or by socioeconomic scale.
# 8. create from the timestamp feature i.e. year, month, day of the week
# 9. create from the time feature a morning noon evening night feature
# 10. normalize the data
# 11. balance the data
# 12. use random forest information gain in order to determine which features ae more important
# 13. create confusion matrix and valuation indicators: precision, recall, accuracy, auc
# 14. integrate mode and median
# 15. determine buy_premium as target variable
# 16. buy_premium has NA values handle it

def date_loading(path):
    """loads the data need to get a path"""


# def data_splitting(data):

def feature_engineering(data):
    """receives the project data and cleans the data"""
    data['Gender'].replace("F", 1, inplace=True)  # replace F to 1
    data['Gender'].replace("M", 0, inplace=True)  # replace M to 0 # unsure if needed can it work with m and f?
    print(train_dataset.head(10))  # in order to verify


def drop_unneeded_columns(data):
    """drop unneeded columns(features): user_Id, Unnamed: 0, ought_premium"""
    data = data.drop(
        columns="User_ID")  # dropping the user_Id column. 0 correlation with all of the features and has no meaning.
    data = data.drop(
        columns="Unnamed: 0")  # dropping the Unnamed: 0 column. 0 correlation with all of the features, it seems that it comes from the csv numbering
    data = data.drop(
        columns="Bought_premium")  # dropping the Bought_premium column there is no need for him the same as Buy_premium

    print(train_dataset.head(10))  # in order to verify


# functions to describe the data
def print_data_unique_values(data):
    """prints the unique values and count of all the feature"""
    for column in data:
        print(data[column].value_counts())


def print_data_summaries(data):
    """prints for each feature summery statistics"""
    for column in data:
        print(data[column].describe(include="all", datetime_is_numeric=True))


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
    upper_corr_mat = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
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


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':


train_dataset = pd.read_csv("ctr_dataset_train.csv")
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
# print(train_dataset.head(10))
# print(how_many_na_col(train_dataset))
# print(how_many_na_col_percentage(train_dataset))
# print_data_unique_values(train_dataset)
# print_data_summaries(train_dataset)
# plot_cor_matrix(train_dataset)
print(len(train_dataset.index))
new_train_dataset = train_dataset.dropna(axis=0, thresh=10)  # axis=0 means rows (1 means columns)
# thresh=10 means that if we have at least 10 non Na values we keep the row can also use subset=["colname1",
# "colname2"] - tells where to check for an valuse - after correlation and informarion gain we can decide
# feature_engineering(train_dataset)
print(len(new_train_dataset.index))




# print(train_dataset[["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
#                "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
#                "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
#                "Color_variations", "Dispatch_loc", "Buy_premium"]].describe(include="all", datetime_is_numeric=True)) # show the dataset 5 statistics
# print(train_dataset[["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
# "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
# "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
# "Color_variations", "Dispatch_loc", "Buy_premium"]].median()) # shows median
# print(train_dataset[["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
# "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
# "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
# "Color_variations", "Dispatch_loc", "Buy_premium"]].mode()) # shows mode
