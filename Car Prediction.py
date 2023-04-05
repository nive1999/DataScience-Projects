#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd


# In[35]:


df=pd.read_csv('car data.csv')
df.head()


# In[36]:


df.shape


# In[37]:


print(df['Seller_Type'].unique())


# In[38]:


print(df['Transmission'].unique())


# In[39]:


print(df['Owner'].unique())


# ### Missing values

# In[40]:


df.isnull().sum()


# In[41]:


df.describe()


# In[42]:


df.columns


# In[43]:


final_dataset=df[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[44]:


final_dataset.head()


# In[45]:


final_dataset['Current_year']=2023


# In[46]:


final_dataset.head()


# In[47]:


final_dataset['No.Year']=final_dataset['Current_year']-final_dataset['Year']


# In[48]:


final_dataset.head()


# In[49]:


final_dataset.drop(['Year','Current_year'],axis=1,inplace=True)


# In[50]:


final_dataset.head()


# ###  Convert the categorical variables to OHE

# In[51]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[52]:


final_dataset.head()


# In[53]:


final_dataset[['Fuel_Type_Diesel','Fuel_Type_Petrol','Seller_Type_Individual','Transmission_Manual']].astype(int)


# In[56]:


import seaborn as sns


# In[57]:


sns.pairplot(final_dataset)


# In[62]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


corr_Mat=final_dataset.corr()


# In[59]:


top_corr_features=corr_Mat.index


# In[65]:


plt.figure(figsize=(20,20))

g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[66]:


final_dataset.head()


# In[67]:


x=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[68]:


x.head()


# In[69]:


y.head()


# ### Feature importance

# In[71]:


from sklearn.ensemble import ExtraTreesRegressor


# In[72]:


model=ExtraTreesRegressor()


# In[73]:


model.fit(x,y)


# In[74]:


print(model.feature_importances_)


# In[76]:


##ploting the important features
feature_imp=pd.Series(model.feature_importances_,index=x.columns)


# In[78]:


feature_imp.nlargest(5).plot(kind='barh')


# In[80]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[81]:


xtrain.shape


# In[82]:


from sklearn.ensemble import RandomForestRegressor


# In[84]:


rf_random=RandomForestRegressor()


# In[86]:


import numpy as np


# In[94]:


#Hyperparameter
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]


# In[95]:


print(n_estimators)


# In[93]:


#RandomizedSearchCV

max_features=['auto','sqrt']#for each split'
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
max_depth=[int(x) for x in np.linspace(5,30,num=6)]#for levels in tree
min_samples_split=[2,5,10,15,100]#min no. of samples req to split a node
min_samples_leaf=[1,2,5,10]#min no. of samples req at each leaf node


# In[96]:


from sklearn.model_selection import RandomizedSearchCV
random_grid={
    'n_estimators':n_estimators,'max_features':max_features,
    
    'max_depth':max_depth,'min_samples_split':min_samples_split,
    
    'min_samples_leaf':min_samples_leaf  
}


# In[97]:


print(random_grid)


# In[98]:


rf=RandomForestRegressor()


# In[100]:


rf_random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[103]:


rf_random.fit(xtrain,ytrain)


# In[104]:


predictions=rf_random.predict(xtest)


# In[112]:


sns.displot(ytest-predictions,bins=20, kde=True, color='blue')


# In[114]:


plt.scatter(ytest,predictions)


# In[117]:


#picle file
import pickle
file=open('Randomforest_regression_model.pkl','wb')
pickle.dump(rf_random,file)


# In[ ]:





# In[ ]:





# In[ ]:




