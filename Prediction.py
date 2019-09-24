# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:17:35 2019

@author: Shriyash Shende
"""

import numpy as np
import pandas as pd
import seaborn as sns

final_train = pd.read_csv('C:\\Users\\Shriyash Shende\\Desktop\\TDM BOX movie\\final_train.csv')
final_test = pd.read_csv('C:\\Users\\Shriyash Shende\\Desktop\\TDM BOX movie\\final_test.csv')

final_train.drop(['Unnamed: 0'], axis = 1, inplace = True)
final_test.drop(['Unnamed: 0'], axis = 1, inplace = True)
final_train.columns
#Data Visualisation
sns.distplot(final_train['revenue'])
sns.distplot(np.log(final_train['revenue']))

sns.distplot(final_train['budget'])
sns.distplot(np.log(final_train['budget'] + 1))
sns.distplot(final_train.budget[final_train.budget != 0] + 1)
sns.distplot(np.log(final_train.budget[final_train.budget != 0] + 1))

sns.distplot(final_train['popularity'])
sns.distplot(np.log(final_train['popularity']))

sns.distplot(final_train['runtime'])

final_train['log_revenue'] = np.log(final_train['revenue'])
final_train.drop(['revenue'], axis =1 , inplace = True)
final_train['log_popularity'] = np.log(final_train['popularity'])
final_train.drop(['popularity'], axis =1 , inplace = True)
final_train['log_budget'] = np.log(final_train.budget[final_train.budget != 0] + 1)
final_train.drop(['budget'], axis = 1, inplace = True)
final_train.isnull().sum()
final_train['log_budget'].describe()
final_train['log_budget'].fillna(16.287021, inplace = True)

sns.distplot(final_train['log_budget'])


from sklearn.model_selection import train_test_split
X = final_train.drop(['revenue'], axis = 1)
Y = final_train['revenue']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


import xgboost
regressor=xgboost.XGBRegressor()

booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
random_cv.fit(X_train,Y_train)
random_cv.best_estimator_
reg = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=2, min_child_weight=1, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)

reg.fit(X_train,Y_train)
Y_pred_train  = reg.predict(X_train)
from sklearn.metrics import r2_score
r2_score(Y_train, Y_pred_train, multioutput='variance_weighted') 

Y_pred_test = reg.predict(X_test)
r2_score(Y_test, Y_pred_test, multioutput='variance_weighted') 

Final_test_revenue = reg.predict(final_test)
pL = pd.DataFrame(Final_test_revenue)
pL.rename(columns={0:'revenue'}, inplace = True)
g = pd.read_csv('C:\\Users\\Shriyash Shende\\Desktop\\TDM BOX movie\\sample_submission.csv')
result = pd.concat([g['id'], pL], axis=1)
result.to_csv('C:\\Users\\Shriyash Shende\\Desktop\\TDM BOX movie\\XGBsubmission.csv')
