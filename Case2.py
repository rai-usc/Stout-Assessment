#!/usr/bin/env python
# coding: utf-8

# Case Study #2<br>
# 
# There is 1 dataset(csv) with 3 years worth of customer orders. There are 4 columns in the csv dataset: index, CUSTOMER_EMAIL(unique identifier as hash), Net_Revenue, and Year.
# For each year we need the following information:
# - Total revenue for the current year
# - New Customer Revenue e.g. new customers not present in previous year only
# - Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year
# - Revenue lost from attrition
# - Existing Customer Revenue Current Year
# - Existing Customer Revenue Prior Year
# - Total Customers Current Year
# - Total Customers Previous Year
# - New Customers
# - Lost Customers
# <br>
# Additionally, generate a few unique plots highlighting some information from the dataset. Are there any interesting observations?
# 

# In[1]:


import pandas as p
import scipy.stats
import numpy as n
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import time
from itertools import permutations
import plotly.express as px
import plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = p.read_csv("casestudy.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[68]:


data.shape


# In[5]:


data['customer_email'].value_counts()


# In[6]:


data['customer_email'] = data['customer_email'].str.strip()


# In[7]:


data['net_revenue'].value_counts()


# In[8]:


data['year'].value_counts()


# ## Total revenue for current year (Assuming 2017 as current year)

# In[59]:


data.groupby('year').net_revenue.sum().iloc[-1]


# ## New Customer Revenue e.g. new customers not present in previous year only

# In[10]:


cust_year_revenue = data.groupby(['customer_email','year']).net_revenue.sum()
cust_year_revenue = cust_year_revenue.reset_index().sort_values(by = ['customer_email','year'])
df_16 = cust_year_revenue[cust_year_revenue['year']==2016]


# In[11]:


df_17 = cust_year_revenue[cust_year_revenue['year']==2017]


# In[12]:


df_16.merge(df_17, on = 'customer_email',how = 'right')


# In[13]:


cust_year_revenue['cust_revenue_prev_yr'] = cust_year_revenue.groupby(['customer_email']).net_revenue.shift()


# In[14]:


cust_year_revenue['cust_revenue_prev_2_yr'] = cust_year_revenue.groupby(['customer_email']).net_revenue.shift(2)


# In[15]:


cust_year_revenue.to_csv("cust_year_revenue.csv")


# In[16]:


cust_year_revenue[cust_year_revenue['customer_email']=='aaafxtkgxo@gmail.com']


# In[17]:


cust_year_revenue[cust_year_revenue['customer_email']=='zzztmtdlbv@gmail.com']


# In[18]:


cust_year_revenue['new_customer_yearwise'] = n.where(~p.isnull(cust_year_revenue['net_revenue'])                                                      & p.isnull(cust_year_revenue['cust_revenue_prev_yr'])                                                     & p.isnull(cust_year_revenue['cust_revenue_prev_2_yr']), 'New','Old')


# In[19]:


new_old_revenue = cust_year_revenue[cust_year_revenue['year']!=2015].groupby(['year','new_customer_yearwise']).net_revenue.sum().unstack()
new_old_revenue


# ## Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year

# In[20]:


new_old_revenue.loc['Growth',:] = new_old_revenue.loc[2017,:] - new_old_revenue.loc[2016,:] 
new_old_revenue


# In[21]:


new_old_revenue.loc['Growth','Old']


# ## Revenue lost from attrition

# In[22]:


# p.crosstab(data['year'],data['customer_email'])
customers_churned_15_17 = p.pivot_table(data = cust_year_revenue ,columns = 'year',index = 'customer_email',aggfunc = n.sum ,values = 'net_revenue')
customers_churned_15_17


# In[23]:


customers_churned_15_17[~p.isnull(customers_churned_15_17[2015])                         & p.isnull(customers_churned_15_17[2016]) &                         p.isnull(customers_churned_15_17[2017])]


# ## Revenue lost from attrition

# In[24]:


#revenue lost due to customers not purchasing back in 2016 and 2017. 
#Assuming they would buy have same amount of purchase in both years

customers_churned_15_17[~p.isnull(customers_churned_15_17[2015]) &                        p.isnull(customers_churned_15_17[2016]) &                         p.isnull(customers_churned_15_17[2017])][2015].sum()*2


# In[25]:


customers_churned_15_17[~p.isnull(customers_churned_15_17[2015]) &                         ~p.isnull(customers_churned_15_17[2016]) &                         p.isnull(customers_churned_15_17[2017])][2016].sum()


# In[26]:


#attrition is when you joined in some year and after that never ordered again in next year
#when the customer has not interacted with or purchased from the company.

##Ways to do??
#2015 mein purchase kiya but 2016 and 2017 mein nahi kiya
#2016 mein purchase kiya but 2017 mein nahi kiya

#purchase per year for customer
#churn year


# In[ ]:





# ## Existing Customer Revenue Current Year

# In[27]:


new_old_revenue.loc[2017,'Old']


# ## Existing Customer Revenue Prior Year

# In[28]:


new_old_revenue.loc[2016,'Old']


# ## Total Customers Current Year

# In[29]:


cust_year_revenue[(cust_year_revenue['year'] == 2017)&                  (~p.isnull(cust_year_revenue['net_revenue']))]['customer_email'].count()


# ## Total Customers Previous Year

# In[30]:


# I assumed that previous customers mean only those customers who did some purchasing in 2016
cust_year_revenue[(cust_year_revenue['year'] == 2016)&                  (~p.isnull(cust_year_revenue['net_revenue']))]['customer_email'].count()


# ## New Customers

# In[62]:


new_old_count = cust_year_revenue[cust_year_revenue['year']!=2015].groupby(['year','new_customer_yearwise']).customer_email.count().unstack()
new_old_count
new_old_count['percent'] = new_old_count['New']/(new_old_count['New']+new_old_count['Old'])*100
new_old_count


# In[32]:


new_old_count['New']


# ## Old Customers

# In[33]:


new_old_count['Old']


# ## Lost Customers

# In[64]:


customers_churned_15_17


# ## Lost Customers

# In[67]:


#Assuming customers which were present in 2015, 2016 but were lost in 2017
customers_churned_15_17[~p.isnull(customers_churned_15_17[2015]) &                         ~p.isnull(customers_churned_15_17[2016]) &                         p.isnull(customers_churned_15_17[2017])].reset_index()['customer_email'].count()

