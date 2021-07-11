#Final Project TAU neural networks.
#By Roi & Yossi D
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
import numpy as np

# def read_data_files(name):
#     test_dataset = pd.read_csv("ctr_dataset_test.csv")
#     train_dataset = pd.read_csv("ctr_dataset_train.csv")
#     pd.set_option('display.max_columns', 30)
#     pd.set_option('display.max_rows', 20)
#
#     print(test_dataset.head(10))
#
#     print('Data files loaded successfully!')

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     read_data_files('self')

train_dataset = pd.read_csv("ctr_dataset_train.csv")
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
# print(train_dataset.head(10))
# print(train_dataset[["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
#                "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
#                "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
#                "Color_variations", "Dispatch_loc", "Bought_premium", "Buy_premium"]].describe()) # show the dataset 5 statistics
#maybe we need more statistic?
correlation_matrix = train_dataset.corr() # creating correlation_matrix
# print(correlation_matrix)
# dataplot=sb.heatmap(correlation_matrix) # creating a heat map of the correlation_matrix
# mp.show() # showing the correlation_matrix
sorted_mat = correlation_matrix.unstack().sort_values()
# Retain upper triangular values of correlation matrix and
# make Lower triangular values Null
upper_corr_mat =correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

# Convert to 1-D series and drop Null values
unique_corr_pairs = upper_corr_mat.unstack().dropna()

# Sort correlation pairs
sorted_mat1 = unique_corr_pairs.sort_values()
print(sorted_mat1)

