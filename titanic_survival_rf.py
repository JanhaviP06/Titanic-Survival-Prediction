import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,accuracy_score,classification_report
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

train_data=pd.read_csv('C:\\Users\\janha\\OneDrive\\Desktop\\ML\\train_csv')

X = train_data.drop(columns =['Survived'] ,axis=1)
Y = train_data['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=42)

rf=RandomForestClassifier(max_depth=15, min_samples_split=15, n_estimators=200)
rf.fit(X_train,Y_train)
rf_y_pred=rf.predict(X_test)
print("Random Forest :")
print(classification_report(Y_test,rf_y_pred))
ac_rf=accuracy_score(Y_test,rf_y_pred)
print("Accuracy Score:",ac_rf)

pickle.dump(rf, open('titanicmodel.pkl','wb'))
model_pk = pickle.load(open('titanicmodel.pkl','rb'))
