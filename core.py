from collections import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score


import numpy as np
import pandas as pd
import preprcessUtil
from preprocess import coreProcess
from xgboostClassifier import xgbClassifier

import preprcessUtil
utils = preprcessUtil.Utils()

clf = xgbClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=2,
        nthread=3,
        eta=0.04,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=1.0,
        colsample_bylevel=0.7,
        min_child_weight=1,
        silent=1,
        num_rounds=700,
        seed=1024
    )



full = pd.read_csv("./data.csv")
full.columns = ['timestamp', 'fnumber', 'temperature', 'comic',
                'music', 'rainfall', 'bookname', 'bookpages']


attributes = full.columns[1:]
skf = StratifiedKFold(n_splits=3, shuffle=True)
data = full[attributes].copy()
utils = preprcessUtil.Utils()
y = full['timestamp'].apply(utils.processTime)
cv_scores = []
i = 0



for train_idx, val_idx in skf.split(data, y):
    i += 1
    X = data.copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train, X_val, feats = coreProcess(X, y_train, train_idx, val_idx)

    clf.fit(X_train, y_train)
    #y_val_pred = clf.predict_proba(X_val)
    y_val_pred = clf.predict(X_val)
    pred =  y_val_pred
    label = y_val.values
    correct = 0
    for j in range(len(pred)):
        if pred[j] == label[j]:
            correct += 1
    precision = float(correct*100.00)/len(pred)
    loss = log_loss(y_val, y_val_pred)

    precision = precision_score(label, pred, average='binary')
    recall = recall_score(label, pred, average='binary')
    f1score = f1_score(label, pred, average='binary')


    print("Iteration-{}    loss: {}\n\t\tPrecision:{} Recall:{} F1:{} %%".format(i, loss, precision, recall, f1score))
    cv_scores.append(loss)


print cv_scores



'''
X_train, X_val, feats = coreProcess(data, y_train, train_idx, val_idx)
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

names=['KNN','LR','NB','Tree','RF','GDBT','SVM']
models=[KNeighborsClassifier(),LogisticRegression(),GaussianNB(),\
        DecisionTreeClassifier(),RandomForestClassifier(),\
        GradientBoostingClassifier(),SVC()]

for name, model in zip(names, models):
    score = cross_val_score(model, X_train, y_train, cv=5)
    print("{}:{},{}".format(name, score.mean(), score))

'''

