#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as p
import numpy as n
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


data=p.read_csv('loans_full_schema.csv')
data.head()


# In[3]:


data.columns


# In[4]:


#we will create a dataframe which will contain all the columns we will be using for machine learning
cleaned=data.copy()


# In[5]:


cleaned.shape


# In[6]:


cleaned=cleaned.drop(['issue_month',
       'loan_status', 'initial_listing_status', 'disbursement_method',
       'balance', 'paid_total', 'paid_principal', 'paid_interest',
       'paid_late_fees','emp_title','emp_length','months_since_90d_late','months_since_last_delinq'\
             ,'verification_income_joint','debt_to_income_joint','num_accounts_120d_past_due','term', 'installment'],axis=1)


# In[7]:


cleaned.shape


# In[8]:


#data.debt_to_income.median()


# In[9]:


#data[~data.annual_income_joint.isna()][['annual_income','annual_income_joint']]


# In[10]:


#cleaned['combined_annual_income']
# n.max(cleaned['annual_income'],cleaned['annual_income_joint'])
#cleaned[['annual_income','annual_income_joint']].max(axis=1)


# In[11]:


cleaned['annual_income']=cleaned[['annual_income','annual_income_joint']].max(axis=1)


# In[12]:


cleaned=cleaned.drop(['annual_income_joint'],axis=1)
cleaned.shape


# In[13]:


cleaned.columns


# In[14]:


# now we split our data into train and test
y=cleaned['interest_rate'].values
X=cleaned.drop(['interest_rate'],axis=1)

train, test = train_test_split(cleaned, test_size=0.2)


# In[15]:


#cleaned['debt_to_income']=cleaned['debt_to_income'].fillna(cleaned['debt_to_income'].median())
train['debt_to_income']=train['debt_to_income'].fillna(train['debt_to_income'].median())
test['debt_to_income']=test['debt_to_income'].fillna(train['debt_to_income'].median())


# In[16]:


train.to_csv('train.csv')


# In[17]:


train['months_since_last_credit_inquiry']=["less than 3" if i<=3         else "between 4 and 8" if i<=8 else "between 9 and 19" if i<=19         else "greater than 19" for i in train['months_since_last_credit_inquiry']]

test['months_since_last_credit_inquiry']=["less than 3" if i<=3         else "between 4 and 8" if i<=8 else "between 9 and 19" if i<=19         else "greater than 19" for i in test['months_since_last_credit_inquiry']]


# In[18]:


median_by_state=train.groupby('state').interest_rate.median()
train['median_of_state']=train.state.map(median_by_state)
test['median_of_state']=test.state.map(median_by_state)


# In[19]:


train['state_categories']=["less than 10.5" if i<10.5 else "between 10.5 and 11" if i<11 else                             "between 11 and 11.5" if i<11.5 else "between 11.5 and 12" if i<12 else                            "between 12 and 13" if i<13 else "more than 13" for i in train['median_of_state']]

test['state_categories']=["less than 10.5" if i<10.5 else "between 10.5 and 11" if i<11 else                             "between 11 and 11.5" if i<11.5 else "between 11.5 and 12" if i<12 else                            "between 12 and 13" if i<13 else "more than 13" for i in test['median_of_state']]


# ## Target encoding

# In[20]:


#cleaned[['state','median_of_state','state_categories']]
# cleaned=cleaned.drop(['state','median_of_state'],axis=1)
# cleaned.info()


# In[21]:


#data.homeownership.value_counts()


# In[22]:


#train.groupby('homeownership').interest_rate.mean()


# In[23]:


#train.groupby('verified_income').interest_rate.mean()


# In[24]:


#cleaned.groupby('months_since_last_credit_inquiry').interest_rate.mean()


# In[25]:


#cleaned.groupby('loan_purpose').interest_rate.mean()


# In[26]:


#cleaned.groupby('grade').interest_rate.mean()


# In[27]:


#cleaned.groupby('sub_grade').interest_rate.mean()


# In[28]:


#cleaned.groupby('state_categories').interest_rate.mean()


# In[29]:


#cleaned.groupby('application_type').interest_rate.mean()


# In[30]:


def target_encoding(feature):
    t=train.groupby(feature).interest_rate.mean()
    train[feature+"_encoded"]=train[feature].map(t)
    test[feature+"_encoded"]=test[feature].map(t)


# In[31]:


target_encoding('homeownership')
target_encoding('verified_income')
target_encoding('months_since_last_credit_inquiry')
target_encoding('loan_purpose')
target_encoding('grade')
target_encoding('sub_grade')
target_encoding('state_categories')
target_encoding('application_type')


# In[32]:


train=train.drop(['homeownership','verified_income','months_since_last_credit_inquiry',                     'loan_purpose','grade','sub_grade','state_categories','application_type','state'],axis=1)
train.info()


# In[33]:


test=test.drop(['homeownership','verified_income','months_since_last_credit_inquiry',                     'loan_purpose','grade','sub_grade','state_categories','application_type','state'],axis=1)
test.info()


# ### Now we will split in train X, train y, test X, test y

# In[34]:


y_train=train['interest_rate'].values
X_train=train.drop(['interest_rate'],axis=1)

y_test=test['interest_rate'].values
X_test=test.drop(['interest_rate'],axis=1)


# In[35]:


X_train.info()


# ### Wrapping techniques

# In[36]:


# clf=RandomForestRegressor(n_estimators=5,n_jobs=-1)
# sfs=SFS(clf,n_features_to_select=35,direction="forward",scoring="neg_mean_squared_error",cv=2,n_jobs=-1)
# sfs.fit(X_train,y_train)


# In[37]:


X_train.columns


# In[38]:


clf=RandomForestRegressor(n_estimators=5,n_jobs=-1)
clf2=LinearRegression()
sfs=SFS2(clf,k_features=35,forward=True,verbose=2,scoring="neg_root_mean_squared_error",cv=2,n_jobs=-1)
sfs.fit(X_train,y_train)


# In[39]:


# import sklearn.metrics
# sklearn.metrics.SCORERS.keys()
sfs.get_metric_dict()


# In[40]:


from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig1 = plot_sfs(sfs.get_metric_dict(),kind='std_dev', figsize=(15, 6))
plt.xticks(n.arange(0, 35, step=1))
plt.ylim([-0.4, 0])
plt.xlim(0,35)
plt.title('Stepwise Selection')
plt.grid()
plt.savefig('FS_fdr.png')
plt.show()


# In[41]:


#sfs.get_support()


# In[42]:


#sfs.transform(X_train)


# In[43]:


sfs.get_metric_dict()[9]['feature_names']


# In[44]:


select=sfs.get_metric_dict()[9]['feature_names']
select


# In[45]:


X_train=X_train[list(select)]
X_test=X_test[list(select)]


# ### Apply machine learning models
# #### Linear Regression

# In[59]:


model=LinearRegression()
model.fit(X_train,y_train)


# In[60]:


pred=model.predict(X_test)


# In[61]:


from math import sqrt
rms = sqrt(mean_squared_error(y_test, pred))
rms


# In[62]:


pred


# In[63]:


y_test


# In[65]:


plt.figure(figsize=(8,8))
plt.scatter(y_test, pred, c='crimson')
# plt.yscale('log')
# plt.xscale('log')

p1 = max(max(pred), max(y_test))
p2 = min(min(pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'g-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.title("Linear Regression Performance")
plt.show()


# #### Random Forest Regressor

# In[66]:


model=RandomForestRegressor(n_estimators=7,n_jobs=-1)
model.fit(X_train,y_train)
pred=model.predict(X_test)


# In[67]:


rms = sqrt(mean_squared_error(y_test, pred))
rms


# In[86]:


plt.figure(figsize=(8,8))
plt.scatter(y_test, pred, c='red')
# plt.yscale('log')
# plt.xscale('log')

p1 = max(max(pred), max(y_test))
p2 = min(min(pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'k-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.title("Random Forest Performance")
plt.show()


# ### SVR

# In[70]:


from sklearn.svm import SVR
model=SVR()
model.fit(X_train,y_train)
pred=model.predict(X_test)


# In[71]:


rms = sqrt(mean_squared_error(y_test, pred))
rms


# In[74]:


plt.figure(figsize=(8,8))
plt.scatter(y_test, pred, c='blue')
# plt.yscale('log')
# plt.xscale('log')

p1 = max(max(pred), max(y_test))
p2 = min(min(pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'g-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.title('Support Vector Regressor Performance')
plt.show()


# In[ ]:





# In[ ]:




