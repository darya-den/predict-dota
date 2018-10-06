# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 22:21:12 2018

@author: Дарья
"""

import pandas
import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

train_data = pandas.read_csv("features.csv", index_col='match_id')
train_y = train_data['radiant_win'] # goal variable
# delete goal variable and features "looking ahead"
train_data = train_data.drop(['duration', 'tower_status_radiant', 'tower_status_dire',
                              'barracks_status_radiant', 'barracks_status_dire', 'radiant_win'], axis=1)
train_data = train_data.fillna(0) # fill in missing data

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# find columns with missing data
#columns_miss = []
#for n, att in enumerate(train_data.count()):
 #   if att < max(train_data.count()):
  #      columns_miss.append(train_data.columns[n])
#print(columns_miss)

n_trees = [5, 10, 15, 20, 25, 30, 35, 40]

# gradient boosting
for nt in n_trees:
    start_time = datetime.datetime.now()
    clf = GradientBoostingClassifier(n_estimators=nt, random_state=42)
    l = cross_val_score(clf, train_data, y=train_y, scoring=make_scorer(roc_auc_score), cv=kf)
    print(nt, l.mean())
    print('Time elapsed: {}'.format(datetime.datetime.now() - start_time))

scaler = StandardScaler()
train_data1 = scaler.fit_transform(train_data)

# regression with all features and C selection
grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf_1 = LogisticRegression(penalty='l2', random_state=42)
gs = GridSearchCV(clf_1, grid, scoring='roc_auc', cv=kf)
start_time = datetime.datetime.now()
gs.fit(train_data1, train_y)
print(gs.best_params_['C'], gs.best_score_)
print('Time elapsed: {}'.format(datetime.datetime.now() - start_time))

# delete categorical features
train_data2 = train_data.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero',
                              'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero',
                              'd4_hero', 'd5_hero'], axis=1)
train_data2 = scaler.fit_transform(train_data2)

# logistic regression without categorical features and C selection
clf_2 = LogisticRegression(penalty='l2', random_state=42)
gs = GridSearchCV(clf_2, grid, scoring='roc_auc', cv=kf)
start_time = datetime.datetime.now()
gs.fit(train_data2, train_y)
print(gs.best_params_['C'], gs.best_score_)
print('Time elapsed: {}'.format(datetime.datetime.now() - start_time))

hero_columns = [x for x in train_data.columns if 'hero' in x]
unique_h = np.unique(train_data[hero_columns]) # нахождение уникальных героев
print(len(unique_h)) # 108

# create "bag-of-words" for categorical features
X_pick = np.zeros((train_data.shape[0], 112))
for i, match_id in enumerate(train_data.index):
    for p in range(5):
        X_pick[i, train_data.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, train_data.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
train_digit = train_data.drop(hero_columns, axis=1)       
train_data_big = np.hstack((train_digit, X_pick)) # объединение признаков
train_data_big = scaler.fit_transform(train_data_big)

# "bag-of-words" regression and C selection
clf_3 = LogisticRegression(penalty='l2', random_state=42)
gs = GridSearchCV(clf_3, grid, refit=True, scoring='roc_auc', cv=kf)
start_time = datetime.datetime.now()
gs.fit(train_data_big, train_y)
print(gs.best_params_['C'], gs.best_score_)
print('Time elapsed: {}'.format(datetime.datetime.now() - start_time))

# test set
test_data = pandas.read_csv("features_test.csv", index_col='match_id')
test_data = test_data.fillna(0)

# "bag-of-words" for test set
X_test = np.zeros((test_data.shape[0], 112))
for i, match_id in enumerate(test_data.index):
    for p in range(5):
        X_test[i, test_data.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_test[i, test_data.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
test_data = np.hstack((test_data.drop(hero_columns, axis=1), X_test))
test_data = scaler.transform(test_data)

# final prediction
prob = gs.predict_proba(test_data)[:,1] 
print(prob, np.amin(prob), np.amax(prob))
