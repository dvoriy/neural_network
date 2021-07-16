# Final Project TAU neural networks.
# By Roi & Yossi D
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



# import pandas_profiling as pp # a libery that helps explore the data
# profile_report = pp.ProfileReport(train_dataset, title="profile report of train data set", minimal=True)
# profile_report.to_file("output.html")

# to do list:
# 1. split the data: train 70%; validation 15%; test 10%
# 6. we need to add one summery statistic to non-numeric variables
# 7. maybe create a function for the location feature. map the locations, organize each town to an area: north, south
#    center or by socioeconomic scale.
# 9. create from the time feature a morning noon evening night feature
# 10. normalize the data
# 11. balance the data
# 12. use random forest information gain in order to determine which features ae more important
# 13. create confusion matrix and valuation indicators: precision, recall, accuracy, auc
# 15. determine buy_premium as target variable
# 16. do we want to turn bought premium to numric and also other variables in order to see better correltion?
# 17. day + month that has a lot of premium create list and if it is one of them create a col of y0 or 1
# 18. train model
# 19. explainable AI

#done
# 2. determine which col if any have to many NA values and therefore are unnecessary
#    meaning that that we leave a feature outside
#    create a function
# 3. determine which rows if any have to many NA values and therefore are unnecessary
#    meaning that that we leave an client outside
#    create a function
# 4. determine how to handle NA values and write a function
# 5. clean the data - look for negative values etc and decided what to do Post_premium_commercial Idle
# 8. create from the timestamp feature i.e. year, month, day of the week
# 14. integrate mode and median in the summery


# def feature_engineering(data):
#     """receives the project data and cleans the data"""
#     data['Gender'].replace("F", 1, inplace=True)  # replace F to 1 # explain why
#     data['Gender'].replace("M", 0, inplace=True)  # replace M to 0 # unsure if needed can it work with m and f?
#     print(train_dataset.head(10))  # in order to verify


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


train_dataset = pd.read_csv("ctr_dataset_train.csv") # loading the data
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Data Exploration
print(train_dataset.head(10)) # print the first 10 rows
for column in train_dataset:
    print(train_dataset[column].value_counts())# prints the unique values in each col and counts them

for column in train_dataset:
    print(train_dataset[column].describe(include="all", datetime_is_numeric=True))# prints the data summaries

# print(train_dataset[["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
#                "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
#                "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
#                "Color_variations", "Dispatch_loc", "Bought_premium", "Buy_premium"]].describe(include="all", datetime_is_numeric=True)) # show the dataset 5 statistics
print(train_dataset[["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
"Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
"Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
"Color_variations", "Dispatch_loc", "Bought_premium", "Buy_premium"]].median()) # shows median
# print(train_dataset[["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
# "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
# "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
# "Color_variations", "Dispatch_loc", "Bought_premium", "Buy_premium"]].mode()) # shows mode

plot_cor_matrix(train_dataset)# creating cor mat for only the  numeric and undropped variables.
# non numeric variables which have no hierarchy - most of the time won't give us meaningful information
# do we want to create cor matrix with the non numeric data????? using hierarchy?
train_dataset = train_dataset.drop(columns="User_ID")  # dropping the user_Id column.
# 0 correlation with all of the features and has no meaning.
train_dataset = train_dataset.drop(columns="Unnamed: 0")  # dropping the Unnamed: 0 column.
# 0 correlation with all of the features, it seems that it comes from the csv numbering

# Fixing negative valuse - we assume that the negative values represent people who didn't Post_premium_commercial
train_dataset[train_dataset['Idle'] < 0] = 0
train_dataset[train_dataset['Post_premium_commercial'] < 0] = 0
# print_data_summaries(train_dataset)

# NA handling
print(len(train_dataset.index)) # number of lines in the data
print(how_many_na_col(train_dataset)) # number of na in each col
print(how_many_na_col_percentage(train_dataset)) # percentage of na in each col

# ניסתי להשתמש בשיטה של IMPUTEולהחליף את הערכים בחציון הבעיה היא שאי אפשר להשתמש בחציון על ערכים קטגוריאלים
# בנוסף שנסיתי להשתמש בהכי נפוץ שעובד על ערכים קטגוראליים לא הצלחתי להשתמש בנתונים נתן שגיאה
# לכן בנתיים השתמשתי בפונקצייה FILLNA שפשוט מללאת את ה NA בערך הקודם או ההבא
# קריטי לתקן את זה אחרת נקבל תוצאות לא טובות
# הכי טוב להשתמש ב KNN
# Color_variations העמודות Commercial_2 ו Commercial_3 ו Size_variations
# יש להן יותר מ 40% ערכים נעלמים אבל בחרתי לא להוריד אותן בנתיים
# בחרתי כן להוריד כל שורה שאין לה משתנה מטרה ידוע מדובר בהורדה של 2608 שורות
# לאחר שנריץ חשיבות משתנים נוכל להחליט להוריד למשל שורות שאין להן את המשתנים החשובים ביותר
# אולי צריך להפריד את הנתונים לנומרי ולא נומרי ואז לבצע את השינוי ולאחר מכן לחבר

# fill NA
missing_values_count = (train_dataset.isnull().sum() * 100 / len(train_dataset))
print(missing_values_count)
train_dataset_filled = train_dataset.fillna(method="ffill")# fill the NA with the priveous value
train_dataset_filled = train_dataset_filled.fillna(method="backfill")# fill the NA with the next value
missing_values_count1 = (train_dataset_filled .isnull().sum() * 100 / len(train_dataset_filled ))
print(missing_values_count1)

# dropping col or rows
# new_train_dataset = train_dataset.dropna(axis=0, thresh=10)  # axis=0 means rows (1 means columns)
# thresh=10 means that if we have at least 10 non Na values we keep the row can also use subset=["colname1",
# "colname2"] - tells where to check for an valuse - after correlation and information gain we can decide
# feature_engineering(train_dataset)

# הורדה של שורות שאין להן משתנה מטרה 2608
print(len(train_dataset.index))
new_train_dataset = train_dataset.dropna(axis=0, thresh=1, subset=["Buy_premium"]) # throws out all the rows with
# NA at the target variable
print(len(new_train_dataset.index))
print(len(train_dataset.index)-len(new_train_dataset.index))

#imputaion
# my_imputer = SimpleImputer(strategy="most_frequent") # we crates an imputer - this object will imput (replace missing values
# # with other values) we chose strategy="most_frequent" but we can choose others as well
# # If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
# # If “median”, then replace missing values using the median along each column. Can only be used with numeric data.
# # If “most_frequent”, then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.
# # If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data.
# data_with_imputed_values = my_imputer.fit_transform(train_dataset)
# missing_values_count_new = (data_with_imputed_values.isnull().sum() * 100 / len(data_with_imputed_values))
# print(missing_values_count_new)


######## feature engineering #########
# Date
train_dataset['Date']= pd.to_datetime(train_dataset['Date'], dayfirst=True) # transform the date to tpe datetime
train_dataset["day"] = train_dataset.apply(lambda row: row.Date.day_name(), axis=1) # creates a new col with day name
# print( train_dataset["day"])
train_dataset["month"] = train_dataset.apply(lambda row: row.Date.month_name(), axis=1) # creates new col with month name
# print( train_dataset["month"])

# days with a lot of premium

# time of day

# location

# soci economical

# creating target variable

# random forest information gain

# Data splitting
# Let's say we want to split the data in 70:15:15 for train:valid:test dataset
train_size=0.7

feature_vector = train_dataset.drop(columns = ['Buy_premium']).copy()
target_variable = train_dataset['Buy_premium']

# In the first step we will split the data in training and remaining dataset to be split later to validation and test
feature_vector_train, feature_vector_to_split, target_variable_train, target_variable_to_split = train_test_split(feature_vector,target_variable, train_size=0.7, random_state=0)
# random_state=0 is important because it will give us the same split data everytime

# Now since we want the valid and test size to be equal (15% each of overall data).
# we have to define test_size=0.5 (that is 50% of remaining data)
feature_vector_valid, feature_vector_test, target_variable_valid, target_variable_test = train_test_split(feature_vector_to_split,target_variable_to_split, test_size=0.5, random_state=0)

print(feature_vector_train.shape), print(target_variable_train.shape)
print(feature_vector_valid.shape), print(target_variable_valid.shape)
print(feature_vector_test.shape), print(target_variable_test.shape)
