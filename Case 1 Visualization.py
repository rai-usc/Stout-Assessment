#!/usr/bin/env python
# coding: utf-8

# emp_title
# Job title.
# 
# emp_length
# Number of years in the job, rounded down. If longer than 10 years, then this is represented by the value 10.
# 
# state
# Two-letter state code.
# 
# home_ownership
# The ownership status of the applicant's residence.
# 
# annual_income
# Annual income.
# 
# verified_income
# Type of verification of the applicant's income.
# 
# debt_to_income
# Debt-to-income ratio.
# 
# annual_income_joint
# If this is a joint application, then the annual income of the two parties applying.
# 
# verification_income_joint
# Type of verification of the joint income.
# 
# debt_to_income_joint
# Debt-to-income ratio for the two parties.
# 
# delinq_2y
# Delinquencies on lines of credit in the last 2 years.
# 
# months_since_last_delinq
# Months since the last delinquency.
# 
# earliest_credit_line
# Year of the applicant's earliest line of credit
# 
# inquiries_last_12m
# Inquiries into the applicant's credit during the last 12 months.
# 
# total_credit_lines
# Total number of credit lines in this applicant's credit history.
# 
# open_credit_lines
# Number of currently open lines of credit.
# 
# total_credit_limit
# Total available credit, e.g. if only credit cards, then the total of all the credit limits. This excludes a mortgage.
# 
# total_credit_utilized
# Total credit balance, excluding a mortgage.
# 
# num_collections_last_12m
# Number of collections in the last 12 months. This excludes medical collections.
# 
# num_historical_failed_to_pay
# The number of derogatory public records, which roughly means the number of times the applicant failed to pay.
# 
# months_since_90d_late
# Months since the last time the applicant was 90 days late on a payment.
# 
# current_accounts_delinq
# Number of accounts where the applicant is currently delinquent.
# 
# total_collection_amount_ever
# The total amount that the applicant has had against them in collections.
# 
# current_installment_accounts
# Number of installment accounts, which are (roughly) accounts with a fixed payment amount and period. A typical example might be a 36-month car loan.
# 
# accounts_opened_24m
# Number of new lines of credit opened in the last 24 months.
# 
# months_since_last_credit_inquiry
# Number of months since the last credit inquiry on this applicant.
# 
# num_satisfactory_accounts
# Number of satisfactory accounts.
# 
# num_accounts_120d_past_due
# Number of current accounts that are 120 days past due.
# 
# num_accounts_30d_past_due
# Number of current accounts that are 30 days past due.
# 
# num_active_debit_accounts
# Number of currently active bank cards.
# 
# total_debit_limit
# Total of all bank card limits.
# 
# num_total_cc_accounts
# Total number of credit card accounts in the applicant's history.
# 
# num_open_cc_accounts
# Total number of currently open credit card accounts.
# 
# num_cc_carrying_balance
# Number of credit cards that are carrying a balance.
# 
# num_mort_accounts
# Number of mortgage accounts.
# 
# account_never_delinq_percent
# Percent of all lines of credit where the applicant was never delinquent.
# 
# tax_liens
# a numeric vector
# 
# public_record_bankrupt
# Number of bankruptcies listed in the public record for this applicant.
# 
# loan_purpose
# The category for the purpose of the loan.
# 
# application_type
# The type of application: either individual or joint.
# 
# loan_amount
# The amount of the loan the applicant received.
# 
# term
# The number of months of the loan the applicant received.
# 
# interest_rate
# Interest rate of the loan the applicant received.
# 
# installment
# Monthly payment for the loan the applicant received.
# 
# grade
# Grade associated with the loan.
# 
# sub_grade
# Detailed grade associated with the loan.
# 
# issue_month
# Month the loan was issued.
# 
# loan_status
# Status of the loan.
# 
# initial_listing_status
# Initial listing status of the loan. (I think this has to do with whether the lender provided the entire loan or if the loan is across multiple lenders.)
# 
# disbursement_method
# Dispersement method of the loan.
# 
# balance
# Current balance on the loan.
# 
# paid_total
# Total that has been paid on the loan by the applicant.
# 
# paid_principal
# The difference between the original loan amount and the current balance on the loan.
# 
# paid_interest
# The amount of interest paid so far by the applicant.
# 
# paid_late_fees
# Late fees paid by the applicant

# In[1]:


import pandas as p
import numpy as n
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:





# In[2]:


data=p.read_csv('loans_full_schema.csv')
data.head()


# In[3]:


data.emp_title.nunique()


# In[4]:


data.emp_title.value_counts()


# In[5]:


data.homeownership.nunique()


# In[6]:


data.emp_title.value_counts().head(20).plot(kind='bar')


# In[7]:


data.homeownership.value_counts()


# In[8]:


data.annual_income.max()


# In[9]:


data.annual_income.min()


# In[10]:


plt.figure(figsize=(12,10))
sns.distplot(data.annual_income)


# In[11]:


data.info()


# In[12]:


data.shape


# In[13]:


data.total_credit_lines


# In[14]:


data.verified_income.value_counts()


# In[15]:


test=data.dropna()


# In[16]:


test.shape


# In[17]:


data.head().transpose()


# In[18]:


sns.distplot(data.interest_rate)


# In[19]:


sns.scatterplot(data.annual_income,data.interest_rate)


# In[20]:


data.debt_to_income.max()


# In[21]:


n.percentile(data.debt_to_income,67)


# In[22]:


data.debt_to_income.quantile(q=0.99)


# In[23]:


sns.boxplot(y=data.debt_to_income)


# In[24]:


sns.distplot(data.delinq_2y)


# In[25]:


data[~data.annual_income_joint.isna()][['annual_income','annual_income_joint']]


# In[26]:


data.loan_amount.min()


# In[27]:


data.public_record_bankrupt.value_counts()


# In[28]:


data[data.public_record_bankrupt==0].interest_rate.mean()


# In[30]:


data[data.public_record_bankrupt==3].interest_rate.mean()


# In[31]:


data.groupby('public_record_bankrupt').interest_rate.mean().plot(kind='bar')


# In[32]:


data.verification_income_joint.value_counts()


# In[33]:


data[~data.verification_income_joint.isna()][['annual_income','annual_income_joint','verified_income','verification_income_joint']]


# In[34]:


data.debt_to_income.mean()


# In[35]:


data.debt_to_income.max()


# In[36]:


sns.boxplot(data.annual_income)


# In[37]:


for i in range(1,101):
    print(data.annual_income.quantile(q=i/100))


# In[38]:


data.annual_income.quantile(q=0.999993)


# In[39]:


data[data.debt_to_income.isna()].info()


# In[40]:


data.debt_to_income.median()


# In[41]:


data.info()


# In[42]:


data.months_since_90d_late.value_counts()


# In[43]:


data[~data.debt_to_income_joint.isna()][['debt_to_income','debt_to_income_joint','verified_income','verification_income_joint']].info()


# In[44]:


data[~data.debt_to_income_joint.isna()][['debt_to_income','debt_to_income_joint','verified_income','verification_income_joint']]


# In[45]:


t=data[~data.debt_to_income_joint.isna()][['debt_to_income','debt_to_income_joint','verified_income','verification_income_joint']]


# In[ ]:


columnList = []
for i in categorical:
    fieldName = i
    nonMissing = df[i].count() * 100 / numrecords
    uniqueValues = len(df[i].unique())
    mode = df[i].mode()[0]
    column = pd.Series({'Field Name':fieldName,'% Populated': nonMissing, '# Unique Values':uniqueValues,'Most Common Value': mode})
    columnList.extend([column])
df_summary_category = pd.DataFrame(columnList).set_index('Field Name').round(2)
df_summary_category['# Unique Values'] = df_summary_category['# Unique Values'].apply('{:,.0f}'.format)
df_summary_category


# In[ ]:


numrecords = len(data)
numeric_columns = ['Amount','Date']
countMissingNumeric = df[numeric_columns].apply(lambda x: x.count() * 100 / numrecords)
countMissingNumeric.name = '% Populated'
zeroProportion = df[numeric_columns].apply(lambda x: len(x[x==0]) * 100 / numrecords)
data_describe = df[['Date','Amount']].describe(datetime_is_numeric=True)
data_describe.loc['mean','Date'] = np.nan
data_describe.loc['min','Date'] = data_describe.loc['min','Date'].strftime('%Y-%m-%d')
data_describe.loc['max','Date'] = data_describe.loc['max','Date'].strftime('%Y-%m-%d')
data_describe.loc['25%','Date'] = data_describe.loc['25%','Date'].strftime('%Y-%m-%d')
data_describe.loc['50%','Date'] = data_describe.loc['50%','Date'].strftime('%Y-%m-%d')
data_describe.loc['75%','Date'] = data_describe.loc['75%','Date'].strftime('%Y-%m-%d')

df_summary_numeric = data_describe.T
df_summary_numeric['% Zero'] = zeroProportion
df_summary_numeric['% Populated'] = df_summary_numeric['count']/numrecords*100
df_summary_numeric = df_summary_numeric.filter(['% Populated','min','max','mean','std','% Zero','25%','50%','75%'])
df_summary_numeric.loc['Amount'] = df_summary_numeric.loc['Amount'].apply('{:,.2f}'.format)
df_summary_numeric


# In[53]:


data.info()
numeric_columns=['emp_length','annual_income','debt_to_income','annual_income_joint',]


# In[55]:


categories=['emp_title','homeownership','state','verified_income','verification_income_joint','loan_purpose',           'application_type','grade','sub_grade','issue_month','loan_status','initial_listing_status',           'disbursement_method']


# In[61]:


z=data.copy()
z=z.drop(categories,axis=1)
numeric_columns=z.columns
z.columns


# In[66]:


columnList = []
for i in categories:
    fieldName = i
    nonMissing = data[i].count() * 100 / numrecords
    uniqueValues = len(data[i].unique())
    mode = data[i].mode()[0]
    column = p.Series({'Field Name':fieldName,'% Populated': nonMissing, '# Unique Values':uniqueValues,'Most Common Value': mode})
    columnList.extend([column])
df_summary_category = p.DataFrame(columnList).set_index('Field Name').round(2)
df_summary_category['# Unique Values'] = df_summary_category['# Unique Values'].apply('{:,.0f}'.format)
df_summary_category


# In[83]:


df_summary_category.to_excel('Categorical Variable DQR.xlsx')


# In[80]:


columnList = []
for i in numeric_columns:
    fieldName = i
    nonMissing = data[i].count() * 100 / numrecords
    #uniqueValues = len(df[i].unique())
    min_val=data[i].min()
    max_val=data[i].max()
    mean_val=data[i].mean()
    std_val=data[i].std()
    zeroProportion = (data[i]==0).sum()/numrecords*100
    Q1=data[i].quantile(q=0.25)
    Q2=data[i].quantile(q=0.5)
    Q3=data[i].quantile(q=0.75)
    Q4=data[i].quantile(q=0.99)
    Q5=data[i].quantile(q=1)
    column = p.Series({'Field Name':fieldName,'% Populated': nonMissing,                         'min':min_val,'max': max_val,'std':std_val,'% zero':zeroProportion,                       '25%':Q1,'50%':Q2,'75%':Q3,'99%':Q4,'100%':Q5})
    columnList.extend([column])
df_summary_numeric = p.DataFrame(columnList).set_index('Field Name').round(2)
#df_summary_category['# Unique Values'] = df_summary_category['# Unique Values'].apply('{:,.0f}'.format)
df_summary_numeric


# In[84]:


df_summary_numeric.to_excel('Numerical Variable DQR.xlsx')


# In[ ]:




