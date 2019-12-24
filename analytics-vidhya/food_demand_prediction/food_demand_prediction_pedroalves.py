#!/usr/bin/env python
# coding: utf-8

# In[184]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# # Starting with importing of data

# In[266]:


train = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Analytics Vidhya\Food Demand Prediction\train.csv')


# In[267]:


meal_info = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Analytics Vidhya\Food Demand Prediction\meal_info.csv')


# In[268]:


center_info = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Analytics Vidhya\Food Demand Prediction\fulfilment_center_info.csv')


# In[269]:


train.head()


# In[270]:


meal_info.head()


# In[271]:


center_info.head()


# In[272]:


train_total = pd.merge(train,meal_info,how = 'left', on = 'meal_id')
train_total = pd.merge(train_total, center_info, how = 'left', on = 'center_id')


# In[273]:


train_total.head()


# In[274]:


train_total.describe()


# ## We dont seem to have NaN values in the dataframe

# In[275]:


train_total.dtypes


# ## Lets see if we can see any correlation in the data in a chart

# ### First we have to transform the categorical variables

# In[276]:


train_total2 = pd.get_dummies(train_total)


# In[277]:


from scipy.stats import pearsonr


# In[278]:


correlations = pd.DataFrame()
columns = []
pearson = []
for col in train_total2.columns:
    corr = pearsonr(train_total2[col],train_total2.num_orders)
    columns.append(col)
    pearson.append(pearsonr(train_total2[col],train_total2.num_orders)[0])
    print('A correlação da coluna {} com a variável destino é {}'.format(col,corr))


# In[279]:


correlations['col'] = columns
correlations['pearson'] = pearson


# In[280]:


correlations.sort_values(by = 'pearson', ascending=False)


# ## There isnt a variable that has a correlation that is out of the ordinary compared to the others, so we are gonna try another method to filter the features

# In[281]:


from sklearn.feature_selection import SelectPercentile, f_regression


# In[282]:


y = train_total2.num_orders
x = train_total2.drop('num_orders', axis = 1)


# In[242]:


percentile = SelectPercentile(f_regression, percentile=80)


# ## Using the cell above we can see that the optimal percentage of features is 80%

# In[203]:


x_new = percentile.fit_transform(x,y)


# In[204]:


cols_selected = percentile.get_support(indices=True)


# In[205]:


x_new = x.iloc[:,cols_selected]


# In[207]:


from sklearn.model_selection import train_test_split


# In[208]:


x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.3, random_state = 42)


# In[209]:


from sklearn import linear_model


# In[210]:


lr = linear_model.LinearRegression()


# In[211]:


lr.fit(x_train,y_train)


# In[212]:


y_lr = lr.predict(x_test)


# In[213]:


from sklearn.metrics import mean_absolute_error


# In[214]:


mae_lr = mean_absolute_error(y_test, y_lr)


# In[215]:


print(mae_lr/train_total2.num_orders.mean())


# ## The MAE for the Linear Model is about 74% of the mean
# ## Lets try another model

# In[283]:


import xgboost as xgb 


# In[222]:


xgb_reg = xgb.XGBRegressor(objective = 'reg:squarederror',alpha = 2, n_estimators = 1000, learning_rate=0.1, max_depth=10 )
xgb_reg.fit(x_train,y_train)


# In[223]:


y_xgb = xgb_reg.predict(x_test)


# In[224]:


mae_xgb = mean_absolute_error(y_test,y_xgb)


# In[225]:


print(mae_xgb/train_total2.num_orders.mean())


# In[226]:


pickle.dump(xgb_reg, open('modelxgb.pickle', 'wb'))


# ## We can see that the xgboost model performs better than the linear model, we are gonna test with the code below whats the optimal number of features

# In[ ]:


perc = [10,20,30,40,50,60,70,80,90,100]
for i in perc:
    percentile = SelectPercentile(f_regression, percentile=i)
    x_new = percentile.fit_transform(x,y)
    cols_selected = percentile.get_support(indices=True)
    x_new = x.iloc[:,cols_selected]
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.3, random_state = 42)
    xgb_reg = xgb.XGBRegressor(objective = 'reg:squarederror',alpha = 2, n_estimators = 1000, learning_rate=0.1, max_depth=10 )
    xgb_reg.fit(x_train,y_train)
    y_xgb = xgb_reg.predict(x_test)
    mae_xgb = mean_absolute_error(y_test,y_xgb)
    print('para {}% das features o erro absoluto médio é {}'.format(i,mae_xgb))


# ### The optimal number of features is 80%, we will try another 2 models, LGBM and CatBoost

# In[245]:


get_ipython().system('pip install catboost')
get_ipython().system('pip install lightgbm')


# In[284]:


import lightgbm as lgb


# In[301]:


percentile = SelectPercentile(f_regression, percentile=80)
x_new = percentile.fit_transform(x,y)
cols_selected = percentile.get_support(indices=True)
x_new = x.iloc[:,cols_selected]
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.3, random_state = 42)
xgb_reg = xgb.XGBRegressor(objective = 'reg:squarederror',alpha = 2, n_estimators = 1000, learning_rate=0.1, max_depth=10 )
xgb_reg.fit(x_train,y_train)
y_xgb = xgb_reg.predict(x_test)
mae_xgb = mean_absolute_error(y_test,y_xgb)
print('MAE XGB = {}'.format(mae_xgb))


# In[302]:


percentile = SelectPercentile(f_regression, percentile=80)
x_new = percentile.fit_transform(x,y)
cols_selected = percentile.get_support(indices=True)
x_new = x.iloc[:,cols_selected]
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.3, random_state = 42)
lgbm_reg = lgb.LGBMRegressor( n_estimators = 1000, learning_rate=0.1, max_depth=10 )
lgbm_reg.fit(x_train,y_train)
y_lgbm = lgbm_reg.predict(x_test)
mae_lgbm = mean_absolute_error(y_test,y_lgbm)
print('MAE LGBM = {}'.format(mae_lgbm))


# In[251]:


import catboost as cb


# ## Trying on the test set

# In[303]:


sample_submission = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Analytics Vidhya\Food Demand Prediction\sample_submission.csv')


# In[304]:


test = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Analytics Vidhya\Food Demand Prediction\test.csv')


# In[305]:


test_total = pd.merge(test,meal_info,how = 'left', on = 'meal_id')
test_total = pd.merge(test_total, center_info, how = 'left', on = 'center_id')


# In[306]:


x_new.columns


# In[307]:


test_total2 = pd.get_dummies(test_total)


# In[312]:


test_total3 = test_total2[['id','center_id', 'checkout_price', 'base_price', 'emailer_for_promotion',
       'homepage_featured', 'city_code', 'region_code', 'op_area',
       'category_Beverages', 'category_Biryani', 'category_Desert',
       'category_Fish', 'category_Other Snacks', 'category_Pasta',
       'category_Rice Bowl', 'category_Salad', 'category_Sandwich',
       'category_Seafood', 'category_Soup', 'category_Starters',
       'cuisine_Continental', 'cuisine_Indian', 'cuisine_Italian',
       'center_type_TYPE_B', 'center_type_TYPE_C']]


# In[310]:


test_total3['num_orders'] = xgb_reg.predict(test_total3.iloc[:,1:])
predict_xgb = test_total3[['id', 'num_orders']]
num = predict_xgb._get_numeric_data()
num[num < 0] = 0
predict_xgb.set_index('id').to_csv('predict_xgbrev1.csv')


# In[313]:


test_total3['num_orders'] = lgbm_reg.predict(test_total3.iloc[:,1:])
predict_lgbm = test_total3[['id', 'num_orders']]
num = predict_lgbm._get_numeric_data()
num[num < 0] = 0
predict_lgbm.set_index('id').to_csv('predict_lgbmrev1.csv')


# In[237]:





# In[238]:





# In[239]:





# In[ ]:




