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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, SMOTENC
import shap
from sklearn import preprocessing


# #onhotencoder works
# print(train_dataset.head(10))
# # enc = preprocessing.OneHotEncoder(handle_unknown="ignore", sparse=False)
# categorical_columns_names = train_dataset.columns[train_dataset.dtypes.apply(lambda c: np.issubdtype(c, object))]
# print(categorical_columns_names)
# categorical_columns_names = categorical_columns_names.drop("Date")
# categorical_columns_names = categorical_columns_names.drop("Time")
# categorical_columns_names = categorical_columns_names.drop("Bought_premium")
#
# # enc.fit(data_frame[["D"]])
# # data_frame = pd.DataFrame((enc.transform(data_frame[categorical_columns_names])))
# # data_frame.columns = enc.get_feature_names(categorical_columns_names)
# # print(data_frame.head(10))
#
# OH_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
# OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_dataset[categorical_columns_names]))
# # OH_cols_valid = pd.DataFrame(OH_encoder.transform(test_data[object_cols]))
# #
# # # Adding column names to the encoded data set.
# OH_cols_train.columns = OH_encoder.get_feature_names(categorical_columns_names)
# # OH_cols_valid.columns = OH_encoder.get_feature_names(object_cols)
# #
# # # One-hot encoding removed index; put it back
# OH_cols_train.index = train_dataset.index
# # OH_cols_valid.index = test_data.index
# #
# # # Remove categorical columns (will replace with one-hot encoding)
# num_X_train = train_dataset.drop(categorical_columns_names, axis=1)
# # num_X_valid = test_data.drop(object_cols, axis=1)
# #
# # # Add one-hot encoded columns to numerical features
# train_dataset = pd.concat([num_X_train, OH_cols_train], axis=1)
# # test_data = pd.concat([num_X_valid, OH_cols_valid], axis=1)
# print(train_dataset.head(10))

# import category_encoders as ce

# import pandas_profiling as pp # a libery that helps explore the data
# profile_report = pp.ProfileReport(train_dataset, title="profile report of train data set", minimal=True)
# profile_report.to_file("output.html")

# to do list:
# 7. maybe create a function for the location feature. map the locations, organize each town to an area: north, south
#    center or by socioeconomic scale.
# 9. create from the time feature a morning noon evening night feature
# 11. balance the data - cant run it
# 12. use random forest information gain in order to determine which features are more important
# 17. day + month that has a lot of premium create list and if it is one of them create a col of y0 or 1
# 18. train NN
# 19. explainable AI
# 20. maybe add a variable that combines shoes clothing and jewlry etc - they have high cor with target variable
# 21. we can encode the categorical data which means that if we have for example a variable with 3 catgories
# small medium and big than we will get 3 columns: small medium and big.
# in the small column there will be a 1 where the row was small and zero if it was medium or big.
# 22. explain why I imputed each feature and why I chose the way I chose

# done:
# 1. split the data: train 70%; validation 15%; test 10%
# 2. determine which col if any have to many NA values and therefore are unnecessary
#    meaning that that we leave a feature outside
#    create a function
# 3. determine which rows if any have to many NA values and therefore are unnecessary
#    meaning that that we leave an client outside
#    create a function
# 4. determine how to handle NA values and write a function
# 5. clean the data - look for negative values etc and decided what to do Post_premium_commercial Idle
# 6. we need to add one summery statistic to non-numeric variables
# 8. create from the timestamp feature i.e. year, month, day of the week
# 10. normalize the data
# 13. create confusion matrix and valuation indicators: precision, recall, accuracy, auc
# 14. integrate mode and median in the summery
# 15. determine buy_premium as target variable

def plot_cor_matrix(data):
    """receives data and plot a heat map of the correlation matrix and a table of the unique features correlation
    returns the table of the unique features correlation"""
    correlation_matrix = data.corr()  # creating correlation_matrix
    # print(correlation_matrix)
    #dataplot = sb.heatmap(correlation_matrix)  # creating a heat map of the correlation_matrix
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


train_dataset = pd.read_csv("ctr_dataset_train.csv")  # loading the data
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Data Exploration
print(train_dataset.head(10))  # print the first 10 rows
print("")
print("Unique values in each col and count")
for column in train_dataset:
    print(train_dataset[column].value_counts())  # prints the unique values in each col and counts them

print("")
print("Data summaries")
for column in train_dataset:
    print(train_dataset[column].describe(include="all", datetime_is_numeric=True))  # prints the data summaries

# print(train_dataset[["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
#                "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
#                "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
#                "Color_variations", "Dispatch_loc", "Bought_premium", "Buy_premium"]].describe(include="all", datetime_is_numeric=True)) # show the dataset 5 statistics
print("")
print("median of numeric")
print(train_dataset[
          ["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
           "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
           "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
           "Color_variations", "Dispatch_loc", "Bought_premium", "Buy_premium"]].median())  # shows median
# print(train_dataset[["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
# "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
# "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
# "Color_variations", "Dispatch_loc", "Bought_premium", "Buy_premium"]].mode()) # shows mode

print("")
print("Data normalization - we chose not to normalize the data because we use Random forest,"
      "and the model doesn't assume normalized data")
# Data normalization - we chose not to normalize the data because we use Random forest,
# and the model doesn't assume normalized data
# scaler = StandardScaler()
# print(train_dataset.head(10))
# num_cols = train_dataset.columns[train_dataset.dtypes.apply(lambda c: np.issubdtype(c, np.number))] # index of numric colums
# num_cols = num_cols.drop("Buy_premium") # dropping Buy_premium from index
# num_cols = num_cols.drop("Unnamed: 0") # dropping Unnamed: 0 from index
# num_cols = num_cols.drop("User_ID") # dropping User_ID from index
# print(num_cols)
# scaler.fit(train_dataset[num_cols])
# scaler.mean_
# train_dataset[num_cols] = scaler.transform(train_dataset[num_cols])
# train_dataset = pd.DataFrame(train_dataset)#, columns=train_dataset.columns
# print(train_dataset.head(10))

print("")
print("creating new date related features")
# Date handling
train_dataset['Date'].fillna(train_dataset['Date'].mode()[0], inplace=True)  # 0.068555 values are Na imputing here
train_dataset['Date'] = pd.to_datetime(train_dataset['Date'], dayfirst=True)  # transform the date to type datetime
train_dataset["day"] = train_dataset.apply(lambda row: row.Date.day_name(), axis=1)  # creates a new col with day name
train_dataset["month"] = train_dataset.apply(lambda row: row.Date.month_name(),
                                             axis=1)  # creates new col with month name
train_dataset["day of year"] = train_dataset.apply(lambda row: str(row.Date.day_of_year),
                                                   axis=1)  # creates new col with month name

print("")

# Time handling

#### categorical variables exploration ####
print("")
print("we now display for each unique value of categorical feature the number of positive Buy_premium")
print("")
print("Gender sorting percentage of positive buy premium)")
gender_df = train_dataset.drop(
    columns=["User_ID", "Unnamed: 0", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1",
             "Commercial_2",
             "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
             "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
             "Color_variations", "Dispatch_loc", "day", "month", "day of year",
             "Bought_premium"])  # dropping all features except for
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
             "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
             "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
             "Color_variations", "Dispatch_loc", "day", "month", "day of year", "Bought_premium"])
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
             "Color_variations", "Dispatch_loc", "day", "month", "day of year", "Bought_premium"])
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
             "Color_variations", "Dispatch_loc", "day", "month", "day of year", "Bought_premium"])
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
             "Color_variations", "Dispatch_loc", "day", "month", "day of year", "Bought_premium"])
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
             "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
             "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
             "Color_variations", "day", "month", "day of year", "Bought_premium"])
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
             "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
             "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
             "Color_variations", "day", "month", "day of year", "Dispatch_loc"])
grouped = Bought_premium_df.groupby("Bought_premium")
VCL = train_dataset["Bought_premium"].value_counts()
GSL = grouped.sum()
GSL["How many"] = VCL
GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]
print(GSL.sort_values(by="percentage"))

print("")
print("day sorting (percentage of positive buy premium)")
day_df = train_dataset.drop(
    columns=["User_ID", "Unnamed: 0", "Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time",
             "Commercial_1", "Commercial_2",
             "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
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
             "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
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
             "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
             "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
             "Color_variations", "day", "month", "Dispatch_loc", "Bought_premium"])
grouped = day_of_year_df.groupby("day of year")
VCL = train_dataset["day of year"].value_counts()
GSL = grouped.sum()
GSL["How many"] = VCL
GSL["percentage"] = GSL["Buy_premium"] / GSL["How many"]
print(GSL.sort_values(by="percentage"))

print("")
print("Plotting correlation matrix")
plot_cor_matrix(train_dataset)  # creating cor mat for only the numeric categorical d
train_dataset = train_dataset.drop(columns="User_ID")  # dropping the user_Id column.
# 0 correlation with all of the features and has no meaning.
train_dataset = train_dataset.drop(columns="Unnamed: 0")  # dropping the Unnamed: 0 column.
# 0 correlation with all of the features, it seems that it comes from the csv numbering
print("")
print("Plotting correlation matrix after dropped some features (explain in comments)")
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

# כדי להשלים נתונים חסרים הכי טוב להשתמש ב KNN
# עמודה עם יותר מ-60% ערכים חסרים היא מיותרת
# Color_variations העמודות Commercial_2 ו Commercial_3 ו Size_variations
# יש להן יותר מ 40% ערכים נעלמים אבל בחרתי לא להוריד אותן בנתיים
# בחרתי כן להוריד כל שורה שאין לה משתנה מטרה ידוע מדובר בהורדה של 2608 שורות
# לאחר שנריץ חשיבות משתנים נוכל להחליט להוריד למשל שורות שאין להן את המשתנים החשובים ביותר
# אולי צריך להפריד את הנתונים לנומרי ולא נומרי ואז לבצע את השינוי ולאחר מכן לחבר

# fill NA
# missing_values_count = (train_dataset.isnull().sum() * 100 / len(train_dataset))
# print(missing_values_count)
# train_dataset_filled = train_dataset.fillna(method="ffill")# fill the NA with the priveous value
# train_dataset_filled = train_dataset_filled.fillna(method="backfill")# fill the NA with the next value
# missing_values_count1 = (train_dataset_filled .isnull().sum() * 100 / len(train_dataset_filled ))
# print(missing_values_count1)

# dropping col or rows
# new_train_dataset = train_dataset.dropna(axis=0, thresh=10)  # axis=0 means rows (1 means columns)
# thresh=10 means that if we have at least 10 non Na values we keep the row can also use subset=["colname1",
# "colname2"] - tells where to check for an valuse - after correlation and information gain we can decide
# feature_engineering(train_dataset)

# הורדה של שורות שאין להן משתנה מטרה 2608
# can also try to knn to predict but risky
print(len(train_dataset.index))
new_train_dataset = train_dataset.dropna(axis=0, thresh=1, subset=[
    "Buy_premium"])  # throws out all the rows with NA at the target variable
print(len(new_train_dataset.index))
print(len(train_dataset.index) - len(new_train_dataset.index))
train_dataset.dropna(axis=0, thresh=1, subset=["Buy_premium"], inplace=True)  # throws out all the rows with
# NA at the target variable
print(len(train_dataset.index))

# impute categorical data with mode
missing_values_count = (train_dataset.isnull().sum() * 100 / len(train_dataset))
print("")
print("displaying Na count for each feature")
print(missing_values_count)

train_dataset['Gender'].fillna(train_dataset['Gender'].mode()[0], inplace=True)  # there is no significant difference
# between Male and Female in order to predict target variable
# consider to drop
train_dataset['Location'].fillna(train_dataset['Location'].mode()[0], inplace=True)
#
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

missing_values_count = (train_dataset.isnull().sum() * 100 / len(train_dataset))
print("")
print("displaying Na count for each feature after NA handling")
print(missing_values_count)

######## feature engineering #########
# data_frame =pd.DataFrame( #code for experimenting
#     {
#         "A": ["1", "1", "35", "67", "67", "3", "1", "344"],
#         "B": [1, 1, 0, 1, 0, 0, 1, 1],
#         "C": ["a", "a", "v", "v", "d", "d", "s", "s"],
#         "D": ["a", "a", "v", "v", "d", "d", "s", "s"]
#     })
# df1 = data_frame.drop(columns=["C","D"])
#
# grouped = data_frame.groupby("A")
#
# print(grouped.sum())
# Date
# train_dataset['Date'] = pd.to_datetime(train_dataset['Date'], dayfirst=True) # transform the date to type datetime
# train_dataset["day"] = train_dataset.apply(lambda row: row.Date.day_name(), axis=1) # creates a new col with day name
# print( train_dataset["day"])
# train_dataset["month"] = train_dataset.apply(lambda row: row.Date.month_name(), axis=1) # creates new col with month name
# print( train_dataset["month"])
# train_dataset["day of year"] = train_dataset.apply(lambda row: str(row.Date.day_of_year), axis=1) # creates new col with month name
# maybe can create list of holidays
# days with a lot of premium
# tried to see if there is any significant diffrence but didn't notice
# droped_df1 = train_dataset.drop(columns=["User_ID","Unnamed: 0","Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
# "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
# "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
# "Color_variations", "Dispatch_loc", "Bought_premium"])
# grouped = droped_df1.groupby("day of year")
# print(grouped.sum())

# droped_df2 = train_dataset.drop(columns=["User_ID","Unnamed: 0","Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
# "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
# "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
# "Color_variations", "Dispatch_loc", "Bought_premium"])
# grouped = droped_df2.groupby("month")
# print(grouped.sum())
#
# droped_df3 = train_dataset.drop(columns=["User_ID","Unnamed: 0","Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
# "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
# "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
# "Color_variations", "Dispatch_loc", "Bought_premium"])
# grouped = droped_df3.groupby("day")
# print(grouped.sum())

# time of day

# location

# soci economical


#### droping categorical data #####
train_dataset = train_dataset.drop(columns="Gender")
train_dataset = train_dataset.drop(columns="Location")
train_dataset = train_dataset.drop(columns="Mouse_activity_1")
train_dataset = train_dataset.drop(columns="Time")
train_dataset = train_dataset.drop(columns="Date")
train_dataset = train_dataset.drop(columns="Mouse_activity_2")
train_dataset = train_dataset.drop(columns="Mouse_activity_3")
train_dataset = train_dataset.drop(columns="Dispatch_loc")
train_dataset = train_dataset.drop(columns="Bought_premium")
train_dataset = train_dataset.drop(columns="day")
train_dataset = train_dataset.drop(columns="month")
train_dataset = train_dataset.drop(columns="day of year")

# Data splitting
# Let's say we want to split the data in 70:15:15 for train:valid:test dataset
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


# Model: Random forest with 10 trees #
# cols = feature_vector_train.columns
# scaler = RobustScaler()
# feature_vector_train = scaler.fit_transform(feature_vector_train)
# feature_vector_valid = scaler.transform(feature_vector_valid)
#
# feature_vector_train = pd.DataFrame(feature_vector_train, columns=[cols])
# feature_vector_valid = pd.DataFrame(feature_vector_valid, columns=[cols])
#
# rfc = RandomForestClassifier(n_estimators=10, random_state=0) # instantiate the classifier
# rfc.fit(feature_vector_train, target_variable_train) # fit the model
# target_variable_prediction_on_train_validation = rfc.predict(feature_vector_valid) # Predict the Test set results
# print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(target_variable_valid, target_variable_prediction_on_train_validation)))
#
# confusion_matrix = confusion_matrix(target_variable_valid, target_variable_prediction_on_train_validation, labels=[1,0]) # create confusion_matrix
# print('Confusion matrix\n\n', confusion_matrix)
# print("TP, FN")
# print("FP, TN")
#
# display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix) #create an onbject to display the confusion_matrix
# display.plot()
#
# print(classification_report(target_variable_valid, target_variable_prediction_on_train_validation)) # create classification_report of varius indicators
#
# print("AUC score:"+str(roc_auc_score(target_variable_valid,target_variable_prediction_on_train_validation)))

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

confusion_matrix = confusion_matrix(target_variable_valid, target_variable_prediction_on_train_validation,
                                    labels=[1, 0])  # create confusion_matrix
print('Confusion matrix\n\n', confusion_matrix)
print("TP, FN")
print("FP, TN")

display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)  # create an onbject to display the confusion_matrix
display.plot()

print(classification_report(target_variable_valid,
                            target_variable_prediction_on_train_validation))  # create classification_report of varius indicators

print("AUC score:" + str(roc_auc_score(target_variable_valid, target_variable_prediction_on_train_validation)))

# Model: Random forest with 200 trees # # the results are not as good as 100 trees and risk of over fitting
# cols = feature_vector_train.columns
# scaler = RobustScaler()
# feature_vector_train = scaler.fit_transform(feature_vector_train)
# feature_vector_valid = scaler.transform(feature_vector_valid)
#
# feature_vector_train = pd.DataFrame(feature_vector_train, columns=[cols])
# feature_vector_valid = pd.DataFrame(feature_vector_valid, columns=[cols])
#
# rfc = RandomForestClassifier(n_estimators=200, random_state=0) # instantiate the classifier
# rfc.fit(feature_vector_train, target_variable_train) # fit the model
# target_variable_prediction_on_train_validation = rfc.predict(feature_vector_valid) # Predict the Test set results
# print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(target_variable_valid, target_variable_prediction_on_train_validation)))
#
# confusion_matrix = confusion_matrix(target_variable_valid, target_variable_prediction_on_train_validation, labels=[1,0]) # create confusion_matrix
# print('Confusion matrix\n\n', confusion_matrix)
# print("TP, FN")
# print("FP, TN")
#
# display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix) #create an onbject to display the confusion_matrix
# display.plot()
#
# print(classification_report(target_variable_valid, target_variable_prediction_on_train_validation)) # create classification_report of varius indicators
#
# print("AUC score:"+str(roc_auc_score(target_variable_valid,target_variable_prediction_on_train_validation)))

print("We did hyperparameter tuning on the number of trees")
print("we tried 10 trees, 100 trees and 200 trees")
print("the best results were with 100 trees")
print("the number of trees is the number of individual decision trees the random forest creates")
print("with low numbers the more we add the better the results will be")
print("but at a certain point the results will start to be worse due too overfitting")

# SHAP Value
print("")
explainer = shap.TreeExplainer(rfc)  # Create object that can calculate shap values
shap_values = explainer.shap_values(feature_vector_valid)  # Calculate Shap values

shap.summary_plot(shap_values[1], feature_vector_valid)
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
      "the higher the shap value the higher the improtance")
print("summary_plot executed")

