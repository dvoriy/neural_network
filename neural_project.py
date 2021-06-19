#Final Project TAU neural networks.
#By Roy & Yossi D
import pandas as pd

def read_data_files(name):
    test_dataset = pd.read_csv("ctr_dataset_test.csv")
    train_dataset = pd.read_csv("ctr_dataset_train.csv")
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 20)

    print (test_dataset.head(10))

    print('Data files loaded successfully!')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    read_data_files('self')
