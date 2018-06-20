# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 20:52:29 2018

@author: amin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 18:14:51 2018

@author: amin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')

train.columns.values

train.dtypes

desc = train.describe()

sales_price_desc = train['SalePrice'].describe()

train.info()


print(train)

#Distribution

plt.hist(train['SalePrice'])

plt.boxplot(train['SalePrice'])

sns.distplot(train['SalePrice'])

plt.scatter(x= train['SaleType'], y = train['SalePrice'])


plt.scatter(x= train['SaleCondition'], y = train['SalePrice'])


plt.scatter(x= train['GarageCars'], y = train['SalePrice'])


plt.scatter(x= train['ScreenPorch'], y = train['SalePrice'])



#boxplot OverallQual

var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


#boxplot for YearBuilt
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


#Correalation Heatmap

# Compute the correlation matrix
corrmat = train.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap
sns.heatmap(corrmat, vmax=.8, square=True);


#Zoomed heatmap style
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();


#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


#Drop Missing Data
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max()



