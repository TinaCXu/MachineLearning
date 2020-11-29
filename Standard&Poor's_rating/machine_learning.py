import pandas as pd
from sklearn import model_selection
import math
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
def feature_engineering(data):
    data = data.drop('FX_RATE', axis=1)
    data = data.drop('ENTITY_NAME', axis=1)
    #data = data.drop('ACCOUNTING_STANDARD', axis=1)
    data = data.drop('CONSOLIDATION_TYPE', axis=1)
    data = data.drop('CURRENCY', axis=1)
    data = data.drop('ID', axis=1)
    data.ACCOUNTING_STANDARD = pd.factorize(data.ACCOUNTING_STANDARD)[0]# string to integer
    data.RATING = pd.factorize(data.RATING)[0]
    print(data.RATING.value_counts())
    return data

def oversample(data):
    for value in data.RATING.unique():
        while len(data[data.RATING == value]) <= 154 :
            new_sample = data[data.RATING == value].sample(n = 1)
            data = data.append(new_sample)
    return data

#---------preprocess data-----------
f = open('data.csv', encoding='utf-8')
data = pd.read_csv(f)
data = feature_engineering(data)
print(data)
#shuffle
data = data.sample(frac=1)
#------------over sample-----------
'''
#divide into test and train
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size, :]
test_data = data.iloc[train_size:, :]
#oversampling train data
train_data = oversample(train_data)
#only oversample train
y = train_data['RATING']
x = train_data.drop('RATING', axis=1)
x_train, _, y_train, _=model_selection.train_test_split(x,y,test_size=0.01)

y = test_data['RATING']
x = test_data.drop('RATING', axis=1)
_, x_test,_ , y_test=model_selection.train_test_split(x,y,test_size=0.99)

print(x_test.shape)
print(y_test.shape)
print(x_train.shape)
print(y_train.shape)
'''
#------------no over sample-----------
y = data['RATING']
x = data.drop('RATING', axis=1)
#print(len(x), len(y))
print(y.value_counts())
x_train, x_test, y_train , y_test=model_selection.train_test_split(x,y,test_size=0.2)
print(x_test.shape)
print(y_test.shape)
print(x_train.shape)
print(y_train.shape)
print(y_train.value_counts(), y_test.value_counts())


#-----------model------------
models = []
models.append(('RandomForestClassifier', RandomForestClassifier(n_estimators=22, criterion='entropy')))
models.append(('BaggingClassifier', BaggingClassifier()))
models.append(('SVM', SVC(probability=True, kernel="rbf", C=1.0, gamma=0.01)))
model_name = []
for name, model in models:
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    model_name.append(name)
    print('model name: ', name)
    print('confusion matrix: \n', confusion_matrix(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    print('\n ',report)
