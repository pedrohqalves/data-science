#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.set_eng_float_format(accuracy=4)


# In[3]:


train = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Kaggle\Ashrae_Energy_Prediction\train.csv')
w_train = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Kaggle\Ashrae_Energy_Prediction\weather_train.csv')


# In[4]:


b_meta = pd.read_csv(r'E:\Users\quadr\Documents\datascience-arquivos\Kaggle\Ashrae_Energy_Prediction\building_metadata.csv')


# # Meter Types
# 
# ## 0 - Electricity
# ## 1 - ChilledWater
# ## 2 - Steam
# ## 3 - Hotwater

# In[5]:


train_total = pd.merge(train,b_meta,how='left', on= 'building_id')


# In[6]:


train_total2 = pd.merge(train_total,w_train, on=['site_id','timestamp'],how='left')


# ### Now that we merged the dataframes, we will drop the timestamp column so it doesnt affect our model

# In[7]:


train_total2=train_total2.drop('timestamp', axis=1)


# In[ ]:


train_total2.head()


# In[ ]:


# To check how much missing values we have
train_total2.count().plot(kind='barh')


# ### The variables that posess the least amount of entries are:
# #### year_built 
# #### floor_count
# #### cloud_coverage
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
train_total2.groupby('primary_use').meter_reading.mean().plot(kind='barh')


# In[ ]:


# Here we can see how site_ids are occupated
train_total2.groupby('primary_use').site_id.mean().plot(kind='bar')


# In[ ]:


# To try to understand more why education and services buildings consume so much energy, 
#lets have a look at site 7 where most of these buildings are

#As expected, site 7 and 11 have lower temperature readings, that would explain why it needs more energy consumption for example heating
# To try to confirm that theory, lets see how energy is distributed in that site, compared to others.
train_total2.groupby('site_id').air_temperature.mean().plot(kind='bar')


# In[ ]:


# As expected, site 7 and 11 have higher meter means, which means they depend more on other types of energy that aren't
# electricity.
train_total2.groupby('site_id').meter.mean().plot(kind='barh')


# In[ ]:


# Seems like Education, Lodging and Technology tend to have higher buildings, 
# that would explain a bit why Education consumes a lot of energy
train_total2.groupby('primary_use').floor_count.mean().plot(kind='bar')


# In[ ]:


# The higher the bar, means the occupation has different energy consumption types
train_total2.groupby('primary_use').meter.mean().plot(kind='bar')


# In[ ]:


# Parking and services provide the biggest size in square feet
train_total2.groupby('primary_use').square_feet.mean().plot(kind='bar')


# In[ ]:


# Education is by far the most common type of building in this dataset
train_total2.groupby('primary_use').building_id.count().plot(kind='bar')


# In[ ]:


# Same here with sites that are very differently distributed when we look at how many buildings each of them have
train_total2.groupby('site_id').building_id.count().plot(kind='barh')


# ### We are gonna fill the empty values in the dataframe to go to the next step, feature engineering

# In[8]:


train_total2.year_built = train_total2.year_built.fillna(train_total2.year_built.mean())
train_total2.cloud_coverage = train_total2.cloud_coverage.fillna(train_total2.cloud_coverage.mean())
train_total2.precip_depth_1_hr = train_total2.precip_depth_1_hr.fillna(0)
train_total2.sea_level_pressure = train_total2.sea_level_pressure.fillna(train_total2.sea_level_pressure.mean())
train_total2.wind_direction = train_total2.wind_direction.fillna(train_total2.wind_direction.mean())
train_total2.wind_speed = train_total2.wind_speed.fillna(0)
train_total2.dew_temperature = train_total2.dew_temperature.fillna(train_total2.dew_temperature.mean())
train_total2.air_temperature = train_total2.air_temperature.fillna(train_total2.air_temperature.mean())


# In[ ]:


train_total2.floor_count.unique()


# #### With floor_count, we can see that there's no building with 0 floors, so we can assume nan means a building that has only base level.
# 
# #### To help modeling we will substitute nan with 0 for floor_count

# In[9]:


train_total2.floor_count = train_total2.floor_count.fillna(0)


# In[ ]:


train_total2.count().plot(kind='barh',figsize=(15,5))


# In[10]:


# Before that we need to get dummies for all the categorical data
train_total2 = pd.get_dummies(train_total2)


# In[11]:


from sklearn.feature_selection import SelectPercentile, f_regression, f_classif


# In[12]:


x = train_total2.drop('meter_reading', axis=1)


# In[13]:


y = train_total2.meter_reading


# In[14]:


percentile = SelectPercentile(f_regression,percentile=30)


# In[15]:


x_eng = percentile.fit_transform(x,y)


# In[16]:


cols_selected = percentile.get_support(indices=True)


# In[17]:


x_eng = x.iloc[:,cols_selected]


# ## We're going to start with a simple linear regression to see how our accuracy goes

# In[21]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x_eng,y,test_size = 0.3, random_state=42)


# In[19]:


from sklearn import linear_model


# In[25]:


lr = linear_model.LinearRegression()


# In[26]:


lr.fit(x_train,y_train)


# In[27]:


y_predic = lr.predict(x_test)


# In[33]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[34]:


msre = np.sqrt(mean_squared_error(y_test,y_predic))
mae = mean_absolute_error(y_test,y_predic)


# In[35]:


print(msre)
print(mae)


# In[39]:


lr.score(x_test,y_test)


# ## Doesnt seem like the linear regression was a good estimate

# In[ ]:




