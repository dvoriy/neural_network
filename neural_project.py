#Final Project TAU neural networks.
#By Roy & Yossi D
import pandas as pd

def read_data_files_train(name):
    #test_dataset = pd.read_csv("ctr_dataset_test.csv")
    train_dataset = pd.read_csv("ctr_dataset_train.csv")
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 20)
    print(train_dataset.head(10))
    print('Data files loaded successfully!')
    return train_dataset

def show_dataframe_statistic(Data_frame):
    Data_frame.describe()

# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
#    read_data_files('self')

train_dataset = pd.read_csv("ctr_dataset_train.csv")
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)
print(train_dataset[["Gender", "Location", "Date", "Time", "Min_prod_time", "Max_prod_time", "Commercial_1", "Commercial_2",
               "Commercial_3", "Mouse_activity_1", "Mouse_activity_2", "Mouse_activity_3", "Jewelry", "Shoes", "Clothing",
               "Home", "Premium", "Premium_commercial_play", "Idle", "Post_premium_commercial", "Size_variations",
               "Color_variations", "Dispatch_loc",
               "Bought_premium", "Buy_premium"]].describe())