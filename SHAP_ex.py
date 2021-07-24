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
from sklearn.metrics import roc_auc_score
import shap

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
data_frame =pd.DataFrame(
    {
        "A": [10, 20, 30, 30, 40, 50, 60, 70],
        "B": [1, 1, 0, 1, 0, 0, 1, 1],
        "C": [32, 523, 23, 42, 1, 31, 188, 23],
    })

feature_vector = data_frame.drop(columns = ['B']).copy()
target_variable = data_frame['B']
feature_vector_train, feature_vector_valid, target_variable_train, target_variable_valid = train_test_split(feature_vector,target_variable, train_size=0.7, random_state=0)

print("Random forest with 100 trees:")
# Model: Random forest with 100 trees # better results then 10 trees
cols = feature_vector_train.columns
scaler = RobustScaler()
feature_vector_train = scaler.fit_transform(feature_vector_train)
feature_vector_valid = scaler.transform(feature_vector_valid)

feature_vector_train = pd.DataFrame(feature_vector_train, columns=[cols])
feature_vector_valid = pd.DataFrame(feature_vector_valid, columns=[cols])

rfc = RandomForestClassifier(n_estimators=100, random_state=0) # instantiate the classifier
rfc.fit(feature_vector_train, target_variable_train) # fit the model
target_variable_prediction_on_train_validation = rfc.predict(feature_vector_valid) # Predict the Test set results
print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(target_variable_valid, target_variable_prediction_on_train_validation)))

confusion_matrix = confusion_matrix(target_variable_valid, target_variable_prediction_on_train_validation, labels=[1,0]) # create confusion_matrix
print('Confusion matrix\n\n', confusion_matrix)
print("TP, FN")
print("FP, TN")

display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix) #create an onbject to display the confusion_matrix
# display.plot()

print(classification_report(target_variable_valid, target_variable_prediction_on_train_validation)) # create classification_report of varius indicators

print("AUC score:"+str(roc_auc_score(target_variable_valid,target_variable_prediction_on_train_validation)))

explainer = shap.TreeExplainer(rfc) # Create object that can calculate shap values
shap_values = explainer.shap_values(feature_vector_valid) # Calculate Shap values

shap.summary_plot(shap_values[1], feature_vector_valid)



