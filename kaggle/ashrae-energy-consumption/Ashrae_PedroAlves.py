#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


pd.set_eng_float_format(accuracy=4)


# In[ ]:


train = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Kaggle\Ashrae_Energy_Prediction\train.csv')
w_train = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Kaggle\Ashrae_Energy_Prediction\weather_train.csv')


# In[ ]:


b_meta = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Kaggle\Ashrae_Energy_Prediction\building_metadata.csv')


# # Meter Types
# 
# ## 0 - Electricity
# ## 1 - ChilledWater
# ## 2 - Steam
# ## 3 - Hotwater

# In[ ]:


train_total = pd.merge(train,b_meta,how='left', on= 'building_id')


# In[ ]:


train_total = pd.merge(train_total,w_train, on=['site_id','timestamp'],how='left')


# ### Now that we merged the dataframes, we will drop the timestamp column so it doesnt affect our model

# In[ ]:


train_total = train_total.drop('timestamp', axis=1)


# In[ ]:


# To check how much missing values we have
train_total.count().plot(kind='barh')


# In[ ]:


# Deleting floor_count and year_built and cloud coverage that have a lot of missing values

train_total = train_total.drop('floor_count', axis=1)
train_total = train_total.drop('year_built', axis=1)
train_total = train_total.drop('cloud_coverage', axis=1)


# ### The variables that posess the least amount of entries are:
# 
# #### precip_depth
# #### sea_level_pressure
# #### wind_direction
# 
# ## -------------------------------------------------------------------------------
# 
# 

# #### We will try to recover those variables using a replacement for empty values
# #### With year_built and climate variables, we will try replacing with the mean as a first try
# #### First we will do a bit of EDA

# In[ ]:


# Seems like Education Buildings consume a lot of Energy, as Services too
train_total.groupby('primary_use').meter_reading.mean().plot(kind='barh')


# In[ ]:


# Here we can see how site_ids are occupated
train_total.groupby('primary_use').site_id.mean().plot(kind='bar')


# In[ ]:


# To try to understand more why education and services buildings consume so much energy, 
#lets have a look at site 7 where most of these buildings are

#As expected, site 7 and 11 have lower temperature readings, that would explain why it needs more energy consumption for example heating
# To try to confirm that theory, lets see how energy is distributed in that site, compared to others.
train_total.groupby('site_id').air_temperature.mean().plot(kind='bar')


# In[ ]:


# As expected, site 7 and 11 have higher meter means, which means they depend more on other types of energy that aren't
# electricity.
train_total.groupby('site_id').meter.mean().plot(kind='barh')


# In[ ]:


# The higher the bar, means the occupation has different energy consumption types
train_total.groupby('primary_use').meter.mean().plot(kind='bar')


# In[ ]:


# Parking and services provide the biggest size in square feet
train_total.groupby('primary_use').square_feet.mean().plot(kind='bar')


# In[ ]:


# Education is by far the most common type of building in this dataset
train_total.groupby('primary_use').building_id.count().plot(kind='bar')


# In[ ]:


# Same here with sites that are very differently distributed when we look at how many buildings each of them have
train_total.groupby('site_id').building_id.count().plot(kind='barh')


# ### We are gonna fill the empty values in the dataframe to go to the next step, feature engineering

# In[ ]:


train_total.precip_depth_1_hr = train_total.precip_depth_1_hr.fillna(0)
train_total.sea_level_pressure = train_total.sea_level_pressure.fillna(train_total.sea_level_pressure.mean())
train_total.wind_direction = train_total.wind_direction.fillna(train_total.wind_direction.mean())
train_total.wind_speed = train_total.wind_speed.fillna(0)
train_total.dew_temperature = train_total.dew_temperature.fillna(train_total.dew_temperature.mean())
train_total.air_temperature = train_total.air_temperature.fillna(train_total.air_temperature.mean())


# In[ ]:


train_total.count().plot(kind='barh',figsize=(15,5))


# In[ ]:


# Before that we need to get dummies for all the categorical data
train_total = pd.get_dummies(train_total)


# In[ ]:


from sklearn.feature_selection import SelectPercentile, f_regression, f_classif


# In[ ]:


x = train_total.drop('meter_reading', axis=1)


# In[ ]:


y = train_total.meter_reading


# In[ ]:


percentile = SelectPercentile(f_regression,percentile=30)


# In[ ]:


x_eng = percentile.fit_transform(x,y)


# In[ ]:


cols_selected = percentile.get_support(indices=True)


# In[ ]:


x_eng = x.iloc[:,cols_selected]


# In[ ]:


x_eng.columns


# ## We're going to start with a simple linear regression to see how our accuracy goes

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_eng,y,test_size = 0.3, random_state=42)


# In[ ]:


from sklearn import linear_model


# In[ ]:


lr = linear_model.LinearRegression()


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


y_lr = lr.predict(x_test)


# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


msre = np.sqrt(mean_squared_error(y_test,y_lr))
mae = mean_absolute_error(y_test,y_lr)


# In[ ]:


print(msre)
print(mae)


# In[ ]:


lr.score(x_test,y_test)


# In[ ]:


get_ipython().system('pip install xgboost')


# ## Trying a XGBoost Model

# In[ ]:


import xgboost as xgb


# In[ ]:


modelxgb = xgb.XGBRegressor(n_estimators=10, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=3)


# In[ ]:


modelxgb.fit(x_train,y_train)


# In[ ]:


y_xgb = modelxgb.predict(x_test)


# In[ ]:


msre_xgb = np.sqrt(mean_squared_error(y_test,y_xgb))
mae_xgb = mean_absolute_error(y_test,y_xgb)
print(msre_xgb)
print(mae_xgb)


# In[ ]:


import pickle


# In[ ]:


pickle.dump(lr, open('lr.pickle', 'wb'))


# In[ ]:


pickle.dump(modelxgb, open('modelxgb.pickle', 'wb'))


# # Going for the test set

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[2]:


b_meta = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Kaggle\Ashrae_Energy_Prediction\building_metadata.csv')


# ### Mantaining only the columns the model will use
# #### 'building_id', 'meter', 'site_id', 'square_feet', 'sea_level_pressure', 'wind_speed', 'primary_use_Education', 'primary_use_Office'

# In[3]:


w_test = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Kaggle\Ashrae_Energy_Prediction\weather_test.csv')


# In[4]:


w_test.columns


# In[5]:


w_test = w_test.drop ('air_temperature', axis = 1)
w_test = w_test.drop ('cloud_coverage', axis = 1)
w_test = w_test.drop ('dew_temperature', axis = 1)
w_test = w_test.drop ('precip_depth_1_hr', axis = 1)
w_test = w_test.drop ('wind_direction', axis = 1)


# In[6]:


w_test.head()


# In[7]:


b_meta.columns


# In[8]:


b_meta_test = b_meta[['site_id', 'building_id', 'primary_use', 'square_feet']]


# In[9]:


b_meta_test = pd.get_dummies(b_meta_test)


# In[10]:


b_meta_test.columns


# ### Mantaining only the columns the model will use
# #### 'building_id', 'meter', 'site_id', 'square_feet', 'sea_level_pressure', 'wind_speed', 'primary_use_Education', 'primary_use_Office'

# In[11]:


b_meta_test = b_meta_test[['site_id','building_id','square_feet','primary_use_Education', 'primary_use_Office']]


# In[12]:


modelxgb = pickle.load(open('modelxgb.pickle','rb'))


# In[13]:


test = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Kaggle\Ashrae_Energy_Prediction\test.csv')


# In[14]:


test.count()


# In[15]:


test = pd.merge(test,b_meta_test,how = 'left', on = 'building_id')
test = pd.merge(test, w_test, how = 'left', on = ['site_id','timestamp'])
test = test.drop('timestamp', axis = 1)


# In[16]:


test.wind_speed = test.wind_speed.fillna(test.wind_speed.mean())
test.sea_level_pressure = test.sea_level_pressure.fillna(test.sea_level_pressure.mean())


# In[17]:


test.head()


# In[18]:


test2 = test[['row_id','building_id', 'meter', 'site_id', 'square_feet', 'sea_level_pressure', 'wind_speed', 'primary_use_Education', 'primary_use_Office']]


# In[19]:


test2.head()


# In[20]:


test2['meter_reading'] = modelxgb.predict(test2.iloc[:,1:])


# In[22]:


predictions = test2[['row_id','meter_reading']]


# In[23]:


predictions = predictions.set_index('row_id')


# In[24]:


predictions.to_csv('submission_pedroalvesxgb.csv')


# In[25]:


predictions.shape


# In[ ]:




