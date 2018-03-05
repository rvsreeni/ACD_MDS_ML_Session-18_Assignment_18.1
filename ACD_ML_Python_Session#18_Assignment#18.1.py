# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:30:08 2018

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split, cross_val_score

boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target
targetsn = (targets - min(targets))/(max(targets) - min(targets))
#print(features.info())
#print(targetsn[500:600])
features['targetsn'] = targetsn
#print(boston.feature_names)
#print(boston.shape())
features['Clas'] = features.targetsn.map(lambda x: 1 if x>0.6 else 0)
#print(features.head())

#predictions from 2 trees
X = features.drop(['targetsn','Clas'],axis=1)
y = features.Clas
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state = 1)
clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=2)
clf.fit(X_train, y_train)
clf2 = DecisionTreeClassifier(random_state=1, max_depth=5)
clf2.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(roc_auc_score(y_test, predictions))
predictions2 = clf2.predict(X_test)
print(roc_auc_score(y_test, predictions2))

#ensemle predictions
predictions = clf.predict_proba(X_test)[:,1]
predictions2 = clf2.predict_proba(X_test)[:,1]
combined = (predictions + predictions2) / 2
rounded = np.round(combined)
print(roc_auc_score(y_test, rounded))