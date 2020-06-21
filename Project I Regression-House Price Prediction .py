#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
import plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')


# # House Price Prediction

# # 1.Data Acquistion

# In[2]:


df = pd.read_csv('../Desktop/DataCamp/HousePricePrediction.csv')
df


# In[3]:


df.shape


# #### There are 4,600 houses in this data set and 18 features including prices.

# # 2.Data Cleaning

# In[4]:


df.isnull().sum()   


# #### No null values in this data set. Let's dive into data analysis.

# # 3.Data Analysis

# In[5]:


df.info()


# #### We have different types of variables.  Most are numerical data eg. price, bedrooms, sqft living and etc. while some are objects eg. date, street and etc. 

# In[6]:


df.describe()


# #### Average price of houses is $550,000

# In[7]:


df['country'].value_counts()


# #### All of houses are in America

# In[8]:


df['statezip'].value_counts()   


# #### All of houses are located in state of Washington, USA. 

# In[9]:


fig = plt.figure(figsize = (12,8))
sns.distplot(df['price'])


# #### Housing price distribution is not a normal distritution.

# In[10]:


fig = plt.figure(figsize = (12,8))
sns.barplot(x = 'condition', y = 'price', data = df)


# In[11]:


fig = plt.figure(figsize = (12,8))
sns.countplot(df['bedrooms'])


# In[12]:


fig = plt.figure(figsize = (12,8))
sns.countplot(df['bathrooms'])


# In[13]:


fig = plt.figure(figsize = (12,8))
sns.countplot(df['floors'])


# #### Majority of houses have condition 3 and 4, 3 to 4 bedrooms, 2.5 bathrooms and 1 to 2 floors.

# In[14]:


fig = plt.figure(figsize = (12,8))
sns.distplot(df['sqft_living'])


# In[15]:


fig = plt.figure(figsize = (12,8))
sns.barplot(x = 'bedrooms', y = 'price', data = df)


# In[16]:


groupbybedroom = df.groupby(['bedrooms']).price.agg([len, min, max])
groupbybedroom


# #### From the table above, there are 3 issues here.
# #### 1. There are 2 houses with no bedrooms. 
# #### 2. Some houses have a price of zero.
# #### 3. The highest price of 3 bedrooms is more than $25,000,000. --> It could be an outlier.

# In[17]:


(df['price'] == 0).value_counts()


# #### There are 49 houses with no prices  (which is around 1%)

# In[18]:


df['price'].sort_values(ascending = False)


# In[19]:


sns.boxplot(df['price'], orient = 'v')


# #### It is clearly that there are  2 outliers here (id 4350 and 4346)

# In[20]:


new_df = df[(df.price < 12000000) & (df.bedrooms > 0) & (df.price>0)]
new_df


# #### Solved 3 problems above by removing them from the original data set --> new_df

# In[21]:


fig = plt.figure(figsize = (12,8))
sns.heatmap(new_df.corr() , annot = new_df.corr())


# In[22]:


fig = plt.figure(figsize = (12,8))
sns.scatterplot(data = df, y = 'sqft_above', x ='sqft_living')


# #### sqft_above and sqft_living have a very high correlation at 0.88. Select only 1 variable

# # 4.Feature Selection

# #### From the correlation table and scatterplot above, I will select all features except the ones that are highly linear correlated with sqft_living 
# #### 1. sqft_living
# #### 2. bedrooms
# #### 3. bathrooms
# #### 4. yr_built
# #### 5. yr_renovated
# #### 6. waterfront
# #### 7. view
# #### 8. condition
# #### 9. floors

# # 5.Model Building

# #### I will build Linear Regression and Support Vector Regression (Linear and rbf) by using both 1 and many independent variables

# ## 1.Simple Linear Regression

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 


# In[24]:


X = new_df['sqft_living']
y = new_df['price']


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)


# In[26]:


X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)


# In[27]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[28]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# In[29]:


print(lm.intercept_)
print(lm.coef_)


# In[30]:


predicted_simple_linear = lm.predict(X_test)
predicted_simple_linear


# In[31]:


print('MAE',metrics.mean_absolute_error(predicted_simple_linear,y_test))
print('MSE',metrics.mean_squared_error(predicted_simple_linear,y_test))
print('RMSE',np.sqrt(metrics.mean_squared_error(predicted_simple_linear,y_test)))


# In[32]:


fig = plt.figure(figsize = (12,8))
plt.scatter(X_test,y_test, color = 'blue',label = 'real price')
plt.plot(X_test,predicted_simple_linear, color = 'red',label = 'predicted regression price')
plt.title('Simple Linear Regression')
plt.xlabel('sqft living')
plt.ylabel('house price')
plt.legend()
plt.xlim([0,7500])
plt.ylim([0,3500000])


# ## 2.Multiple Linear Regression

# In[33]:


X = new_df[['sqft_living','bedrooms','bathrooms','yr_built','yr_renovated','waterfront','view','condition','floors']]
y = new_df['price']


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 100)


# In[35]:


y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)


# In[36]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[37]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# In[38]:


print(lm.intercept_)
print(lm.coef_)


# In[39]:


predicted_multi_linear = lm.predict(X_test)
predicted_multi_linear


# In[40]:


print('MAE',metrics.mean_absolute_error(predicted_multi_linear,y_test))
print('MSE',metrics.mean_squared_error(predicted_multi_linear,y_test))
print('RMSE',np.sqrt(metrics.mean_squared_error(predicted_multi_linear,y_test)))


# ## 3.SVR Linear (1 independent variable)

# In[41]:


X = new_df['sqft_living']
y = new_df['price']


# In[42]:


X = np.array(X).reshape(-1,1)
y = np.array(y).reshape(-1,1)


# In[43]:


from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
sc_y = StandardScaler()

X1 = sc_X.fit_transform(X)
y1 = sc_y.fit_transform(y)


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.2, random_state = 100)


# In[45]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[46]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train ,y_train)


# In[47]:


predicted_SVR_simple_linear = sc_y.inverse_transform(regressor.predict(X_test))
predicted_SVR_simple_linear


# In[48]:


print('MAE', metrics.mean_absolute_error(predicted_SVR_simple_linear,sc_y.inverse_transform(y_test)))
print('MSE', metrics.mean_squared_error(predicted_SVR_simple_linear,sc_y.inverse_transform(y_test)))
print('RMSE', np.sqrt(metrics.mean_squared_error(predicted_SVR_simple_linear,sc_y.inverse_transform(y_test))))


# In[49]:


fig = plt.figure(figsize = (12,8))
plt.scatter(sc_X.inverse_transform(X_test),sc_y.inverse_transform(y_test), color = 'red',label = 'real price')
plt.plot(sc_X.inverse_transform(X_test),predicted_SVR_simple_linear,color = 'blue',label = 'regression price')
plt.title('SVR Linear')
plt.legend()
plt.xlabel('sqft living')
plt.ylabel('Housing Price')
plt.show()


# ## 4.SVR Linear (many independent variables )

# In[50]:


X = new_df[['sqft_living','bedrooms','bathrooms','yr_built','yr_renovated','waterfront','view','condition','floors']]
y = new_df['price']


# In[51]:


y = np.array(y).reshape(-1,1)


# In[52]:


from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
sc_y = StandardScaler()

X2 = sc_X.fit_transform(X)
y2 = sc_y.fit_transform(y)


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.2, random_state = 100)


# In[54]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train,y_train)


# In[55]:


predicted_SVR_multi_linear = sc_y.inverse_transform(regressor.predict(X_test))
predicted_SVR_multi_linear


# In[56]:


print('MAE', metrics.mean_absolute_error(predicted_SVR_multi_linear,sc_y.inverse_transform(y_test)))
print('MSE', metrics.mean_squared_error(predicted_SVR_multi_linear,sc_y.inverse_transform(y_test)))
print('RMSE', np.sqrt(metrics.mean_squared_error(predicted_SVR_multi_linear,sc_y.inverse_transform(y_test))))


# ## 5.SVR rbf ( 1 independent variable)

# In[57]:


X = new_df['sqft_living']
y = new_df['price']


# In[58]:


X = np.array(X).reshape(-1,1)
y = np.array(y).reshape(-1,1)


# In[59]:


from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
sc_y = StandardScaler()

X3 = sc_X.fit_transform(X)
y3 = sc_y.fit_transform(y)


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size = 0.2, random_state = 100)


# In[61]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)


# In[62]:


predicted_SVR_simple_rbf = sc_y.inverse_transform(regressor.predict(X_test))
predicted_SVR_simple_rbf


# In[63]:


print('MAE', metrics.mean_absolute_error(predicted_SVR_simple_rbf,sc_y.inverse_transform(y_test)))
print('MSE', metrics.mean_squared_error(predicted_SVR_simple_rbf,sc_y.inverse_transform(y_test)))
print('RMSE', np.sqrt(metrics.mean_squared_error(predicted_SVR_simple_rbf,sc_y.inverse_transform(y_test))))


# ## 6.SVR rbf ( many independent variables)

# In[64]:


X = new_df[['sqft_living','bedrooms','bathrooms','yr_built','yr_renovated','waterfront','view','condition','floors']]
y = new_df['price']


# In[65]:


y = np.array(y).reshape(-1,1)


# In[66]:


from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
sc_y = StandardScaler()

X4 = sc_X.fit_transform(X)
y4 = sc_y.fit_transform(y)


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size = 0.2, random_state = 100)


# In[68]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)


# In[69]:


predicted_SVR_multi_rbf = sc_y.inverse_transform(regressor.predict(X_test))
predicted_SVR_multi_rbf


# In[70]:


print('MAE', metrics.mean_absolute_error(predicted_SVR_multi_rbf,sc_y.inverse_transform(y_test)))
print('MSE', metrics.mean_squared_error(predicted_SVR_multi_rbf,sc_y.inverse_transform(y_test)))
print('RMSE', np.sqrt(metrics.mean_squared_error(predicted_SVR_multi_rbf,sc_y.inverse_transform(y_test))))


# # 6.Model Evaluation

# #### One independent variable

# In[71]:


print('RMSE Simple Linear Regression', np.sqrt(metrics.mean_squared_error(predicted_simple_linear,sc_y.inverse_transform(y_test))))
print('RMSE SVR Linear', np.sqrt(metrics.mean_squared_error(predicted_SVR_simple_linear,sc_y.inverse_transform(y_test))))
print('RMSE SVR rbf', np.sqrt(metrics.mean_squared_error(predicted_SVR_simple_rbf,sc_y.inverse_transform(y_test))))


# #### Many independent variables

# In[72]:


#Many independent - One dependent 
print('RMSE Multiple Linear Regression', np.sqrt(metrics.mean_squared_error(predicted_multi_linear,sc_y.inverse_transform(y_test))))
print('RMSE SVR Linear', np.sqrt(metrics.mean_squared_error(predicted_SVR_multi_linear,sc_y.inverse_transform(y_test))))
print('RMSE SVR rbf', np.sqrt(metrics.mean_squared_error(predicted_SVR_multi_rbf,sc_y.inverse_transform(y_test))))


# # 7. Suggestion

# #### Linear regression results are better than those of SVR. It is also clearly that using many independent variables have got better results than one independent variable. So I would use multiple Linear Regression to predict the price of a home.
# 
# #### I would take these variables ('sqft_living','bedrooms','bathrooms','yr_built','yr_renovated','waterfront','view','condition','floors') into accounts rather than all variables due to the fact that some predictor variables are highly linearly correlated that brings multicollinearity problem.
# 
# #### NOTE: I used all variables to build both Linear Regression and SVR models and found that the errors were higher  ( no coding here). Please let me know if you are interested to see.
