# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 09:18:57 2018

@author: Administrator
"""


'''
rf:0.12836
lasso:
adboost: 0.41471
gbdt:0.13519

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#加载数据
train = pd.read_csv('./input/train.csv',index_col=0)
test = pd.read_csv('./input/test.csv',index_col=0)

#查看y的分布情况
y = train.pop('SalePrice')
log_y = np.log1p(y)

#画图显示
fig= plt.figure(figsize=(8,5))

ax1 = plt.subplot2grid([1,2],[0,0])  #不做log处理
ax1.hist(y)

ax2 = plt.subplot2grid([1,2],[0,1])  #做log处理
ax2.hist(log_y)

#合并train和test
data = pd.concat([train,test],axis=0)
data['MSSubClass'] = data['MSSubClass'].astype(str)   #这个特征转字符型
#one_hot编码
data_dummy = pd.get_dummies(data)
#有部分缺失值
data_dummy.isnull().sum().sort_values(ascending=False).head(15)

data_mean = data_dummy.mean()
data_dummy.fillna(data_mean,inplace=True)   #均值填充

#标准化那些数据
data_dummy.loc[:,data.columns !='object'] = (data_dummy.loc[:,data.columns !='object'] - data_dummy.loc[:,data.columns !='object'].mean())/data_dummy.loc[:,data.columns !='object'].std()
#划分出train_data,test_date
train_data = data_dummy.loc[train.index]
test_data = data_dummy.loc[test.index]

#建立模型
#========================================

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

#1/lr
lasso = Lasso()
model_lasso = GridSearchCV(lasso,param_grid={'alpha':np.logspace(-3,2,100)},cv=5)
model_lasso.fit(train_data,log_y)
print('最好的参数：',model_lasso.best_estimator_)   
print('得分是：',model_lasso.best_score_)    
lasso_y = np.expm1(model_lasso.predict(test_data))

'''
最好的参数： Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
得分是： 0.8768867547599901
'''

#2/RF
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
model_rf= GridSearchCV(rf,param_grid={'n_estimators':[30,60,90,100],'max_depth':np.arange(3,10)},cv=5)
model_rf.fit(train_data,log_y)
print('最好的参数：',model_rf.best_estimator_)
print('得分是：',model_rf.best_score_)
lr_y = np.expm1(model_rf.predict(test_data))

'''
最好的参数： RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=9,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=90, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
得分是： 0.8683524221007549
'''

#3/boosting
from sklearn.ensemble import AdaBoostRegressor
adboost = AdaBoostRegressor(base_estimator = lasso)
model_adboost = GridSearchCV(adboost,param_grid={'learning_rate':np.logspace(-3,-1,20),'n_estimators':[20,40,60,80,100]})
model_adboost.fit(train_data,log_y)
print('最好的参数：',model_adboost.best_estimator_)
print('得分是：',model_adboost.best_score_)
adboost_y = np.expm1(model_adboost.predict(test_data))

'''
最好的参数： AdaBoostRegressor(base_estimator=Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False),
         learning_rate=0.1, loss='linear', n_estimators=100,
         random_state=None)
得分是： 0.017396793490224997
'''

#4/gbdt
from sklearn.ensemble import GradientBoostingRegressor
gbdt = GradientBoostingRegressor()
model_gbdt= GridSearchCV(gbdt,param_grid={'learning_rate':np.logspace(-3,-1,20),'n_estimators':[20,40,60,80,100]})
model_gbdt.fit(train_data,log_y)
print('最好的参数：',model_gbdt.best_estimator_)
print('得分是：',model_gbdt.best_score_)
gbdt_y = np.expm1(model_gbdt.predict(test_data))
'''
最好的参数： GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
得分是： 0.8942227440070227
'''
#5/xgboost
from xgboost import XGBRegressor
xgb = XGBRegressor()
model_xgb = GridSearchCV(xgb,param_grid=({'n_estimators':[20,40,60,80,100],'learning_rate':np.logspace(-3,-1,20)}))
model_xgb.fit(train_data,log_y)
print('最好的参数：',model_xgb.best_estimator_)
print('得分是：',model_xgb.best_score_)
xgb_y = model_xgb.predict(test_data)
'''
最好的参数： XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
得分是： 0.8885704897402684
'''

#输出预测结果
output_lasso = pd.DataFrame({'Id':test_data.index,'SalePrice':lasso_y})
output_lasso.to_csv('lasso.csv',index=None)

output_rf = pd.DataFrame({'Id':test_data.index,'SalePrice':lr_y})
output_rf.to_csv('lr.csv',index=None)

output_adaboost = pd.DataFrame({'Id':test_data.index,'SalePrice':adboost_y})
output_adaboost.to_csv('adboost_y.csv',index=None)

output_gbdt = pd.DataFrame({'Id':test_data.index,'SalePrice':gbdt_y})
output_gbdt.to_csv('gbdt_y.csv',index=None)

output_xgb = pd.DataFrame({'Id':test_data.index,'SalePrice':xgb_y})
output_xgb.to_csv('xgb_y.csv',index=None)
