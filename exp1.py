import pandas as pd
import tensorflow as tf

train_dataset = pd.read_csv("ctr_dataset_train.csv") # loading the data
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

FIELD_DEFAULTS = [[0], [0], [0], [0], [0],
                  [0], [0], [0], [0], [0],
                  [0], [0], [0], [0], [0],
                  [0], [0], [0], [0], [0],
                  [0], [0], [0], [0], [0],
                  [0], [0]]

feature_names = ["line_num","User_ID","Gender","Location","Date","Time","Min_prod_time","Max_prod_time",
                  "Commercial_1","Commercial_2","Commercial_3","Mouse_activity_1","Mouse_activity_2","Mouse_activity_3",
                "Jewelry","Shoes","Clothing","Home","Premium","Premium_commercial_play","Idle","Post_premium_commercial",
                  "Size_variations","Color_variations","Dispatch_loc","Bought_premium","Buy_premium"]
def parse_line(line):
   parsed_line = tf.io.decode_csv(line, FIELD_DEFAULTS)
   label = parsed_line[-1]
   del parsed_line[-1]
   features = parsed_line
   d = dict(zip(feature_names, features))
   print ("dictionary", d, " label = ", label)
   return d, label

def csv_input_fn(csv_path, batch_size):
   dataset = tf.data.TextLineDataset(csv_path)
   dataset = dataset.map(parse_line)

   dataset = dataset.shuffle(1000).repeat().batch(batch_size)
   return dataset


if __name__ == '__main__':
#    ds = csv_input_fn("ctr_dataset_train.csv", 1)
    Min_prod_time = tf.feature_column.numeric_column("Min_prod_time")
    Max_prod_time = tf.feature_column.numeric_column("Max_prod_time")
    Commercial_1 = tf.feature_column.numeric_column("Commercial_1")
    Commercial_2 = tf.feature_column.numeric_column("Commercial_2")
    Commercial_3 = tf.feature_column.numeric_column("Commercial_3")
    Jewelry = tf.feature_column.numeric_column("Jewelry")
    Shoes = tf.feature_column.numeric_column("Shoes")
    Clothing = tf.feature_column.numeric_column("Clothing")
    Idle = tf.feature_column.numeric_column("Idle")
    Post_premium_commercial = tf.feature_column.numeric_column("Post_premium_commercial")
    Premium_commercial_play = tf.feature_column.numeric_column("Premium_commercial_play")
    Size_variations = tf.feature_column.numeric_column("Size_variations")
    Color_variations = tf.feature_column.numeric_column("Color_variations")
    feature_columns = [Min_prod_time, Max_prod_time, Commercial_1, Commercial_2, Commercial_3, Jewelry, Shoes,
                   Clothing, Idle, Post_premium_commercial, Premium_commercial_play, Size_variations, Color_variations]

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=3,
        model_dir="/tmp")
    batch_size = 100

    classifier.train(
        steps=1000,
        input_fn=lambda: csv_input_fn("ctr_dataset_train.csv", batch_size))
