#Training from: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#The dataset provides the bank customers’ information.
# It includes 41,188 records and 21 fields.
import os
folder = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))
data = pd.read_csv(folder+'/banking.csv')
#PRint data shape
print(data.shape)
print(list(data.columns))
#The education column of the dataset has many categories and we need to reduce
# the categories for a better modelling. The education column has the
# following categories:
#Let us group “basic.4y”, “basic.9y” and “basic.6y” together and call them “basic”.
data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

data['education'].unique()
#Data exploration
print(data['y'].value_counts())
count_no_sub = len(data[data['y']==0])
count_sub = len(data[data['y']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)

#Our classes are imbalanced, and the ratio of no-subscription to
#subscription instances is 89:11. Before we go ahead to balance the classes,
#let’s do some more exploration.
data.describe()
print(data.groupby('y').mean())
#matplotlib inline
pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')

#The frequency of purchase of the deposit depends a great deal on the job title.
# Thus, the job title can be a good predictor of the outcome variable.
table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')

#The marital status does not seem a strong predictor for the outcome variable.
table=pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('edu_vs_pur_stack')
#Education seems a good predictor of the outcome variable.
pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')
#Day of week may not be a good predictor of the outcome.
pd.crosstab(data.month,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_month_bar')
#Month might be a good predictor of the outcome variable.
data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')
pd.crosstab(data.poutcome,data.y).plot(kind='bar')

plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_pout_bar')
#Poutcome seems to be a good predictor of the outcome variable.
#Create dummy variables
#That is variables with only two values, zero and one.
#Lo que hace es para las variables que se pueden separar, crea variables adicionales
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]
data_final.columns.values
print(data_final.columns.values)
#Over-sampling using SMOTE
#With our training data created, I’ll up-sample the no-subscription using the SMOTE
# algorithm(Synthetic Minority Oversampling Technique). At a high level, SMOTE:
#Works by creating synthetic samples from the minor class (no-subscription) instead of
# creating copies.
#Randomly choosing one of the k-nearest-neighbors and using it to create a similar,
# but randomly tweaked, new observations.
#We are going to implement SMOTE in Python.
X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))
'''
Now we have a perfect balanced data! 
You may have noticed that I over-sampled only on the training data, 
because by oversampling only on the training data, 
none of the information in the test data is being used to 
create synthetic observations, therefore, no information will bleed 
from test data into the model training.
'''
'''Recursive Feature Elimination
Recursive Feature Elimination (RFE) is based on the idea to repeatedly 
construct a model and choose either the best or worst performing feature, 
setting the feature aside and then repeating the process with the rest of 
the features. This process is applied until all features in the dataset 
are exhausted. The goal of RFE is to select features by recursively 
considering smaller and smaller sets of features.
'''
data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=200)
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
#Como al ejecutar esto protesta, dice que tengo que escalar
#los datos. Eso intento:
scaler = preprocessing.StandardScaler().fit(os_data_X)
print("average: ", scaler.mean_)
print("scale: ", scaler.scale_)
os_data_X_scaled = scaler.transform(os_data_X)
logreg = LogisticRegression(max_iter=200)

rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X_scaled, os_data_y.values.ravel())

print(rfe.support_)
print(rfe.ranking_)
columns_names= list(os_data_X.columns.values.tolist())
index = 0
columns_selected = []
for column in columns_names:
    if rfe.support_[index]:
        columns_selected.append(column)
    index +=1
print(columns_selected)

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())