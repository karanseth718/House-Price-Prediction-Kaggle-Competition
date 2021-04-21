# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 23:53:37 2021

@author: karan
"""

## Importing Library 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

######################## importing dataset############################

train= pd.read_csv("train.csv")
test= pd.read_csv("test.csv")
train.head()


"""****************Data Exploration*****************""""

#### correlation of SalesPrice with other feature######

corr_matrix = train.corr()
corr_matrix = corr_matrix['SalePrice'].sort_values(ascending = False)
corr_matrix

##### check the distribution SalePrice#####
sns.distplot(train["SalePrice"],kde = False,bins = 10)

"""******************* Dealing with Missing Values***************"""
train.isna().any()
sum(train.isna().any())

######lets join both train and test data set.... so, that we can perform EDA together on both datasets

final_df = pd.concat((train,test)).reset_index(drop = True)

##Seperate targate variable from feauture. So,that we can start filling missing data
final_df.drop(["SalePrice"],axis=1,inplace = True)

#####Check the number of missing values in feature coloumn and their datatype
total = final_df.isnull().sum().sort_values(ascending = False)[final_df.isnull().sum().sort_values(ascending = False) != 0]
data_type = final_df.dtypes.sort_values(ascending = False)[final_df.isnull().sum().sort_values(ascending = False) != 0]
Missing_val_matrix =pd.concat([total, data_type], axis=1, keys=['Total','Type'])

#### On noticing above matrix, we notice that few feature have missing values are really missing. That means that specific huse doen't have that feature
real_missing_feature = ("PoolQC","MiscFeature","Alley","Fence","FireplaceQu","LotFrontage","GarageQual","GarageFinish","GarageYrBlt","GarageCond","GarageType","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","BsmtQual")
for i in real_missing_feature :
    final_df[i]= final_df[i].fillna("None")
    
#Some features consist of numbers that are actually categories so we'll convert to str so they get binarized later

numerical_categorie =('OverallCond','OverallQual','MSSubClass','YearBuilt','YearRemodAdd','GarageYrBlt','YrSold','MoSold')

for j in numerical_categorie:
    final_df[j] = final_df[j].astype(str)
    
###lets check missing values one more time
total1 = final_df.isnull().sum().sort_values(ascending = False)[final_df.isnull().sum().sort_values(ascending = False) != 0]
data_type1 = final_df.dtypes.sort_value


final_df['Functional'] = final_df['Functional'].fillna('Typical')              #Functional: Home functionality (Assume typical unless deductions are warranted) 
final_df['LotFrontage'] = final_df['LotFrontage'].fillna(0)                    #if data ismissing, there is none ... I guess 
final_df['MasVnrArea'] = final_df['MasVnrArea'].fillna(0) 
final_df['Utilities'] = final_df['Utilities'].fillna('AllPub')                 #default values 
final_df['BsmtFullBath'] = final_df['BsmtFullBath'].fillna(0) 
final_df['BsmtHalfBath'] = final_df['BsmtHalfBath'].fillna(0) 
final_df['BsmtFinSF1'] = final_df['BsmtFinSF2'].fillna(0) 
final_df['BsmtFinSF2'] = final_df['BsmtFinSF2'].fillna(0) 
final_df['BsmtUnfSF'] = final_df['BsmtUnfSF'].fillna(0) 
final_df['TotalBsmtSF'] = final_df['TotalBsmtSF'].fillna(0) 
final_df['SaleType'] = final_df['SaleType'].fillna("Oth") 
final_df['GarageCars'] = final_df['GarageCars'].fillna(0) 
final_df['GarageYrBlt'] = final_df['GarageYrBlt'].fillna(0) 
final_df['Electrical'] = final_df['Electrical'].fillna('SBrkr') #default values 
final_df['GarageArea'] = final_df['GarageArea'].fillna(0) 
final_df['Exterior1st'] = final_df['Exterior1st'].fillna(final_df['Exterior1st'].mode()[0])                             #filling with mode 
final_df['Exterior2nd'] = final_df['Exterior2nd'].fillna(final_df['Exterior2nd'].mode()[0]) 
final_df['KitchenQual'] = final_df['BsmtFinSF1'].fillna(final_df['Exterior2nd'].mode()[0])
final_df['MSZoning'] = final_df['MSZoning'].fillna(final_df['MSZoning'].mode()[0])
final_df['MasVnrType'] = final_df['MasVnrType'].fillna(final_df['MasVnrType'].mode()[0])


total = final_df.isnull().sum().sort_values(ascending = False)[final_df.isnull().sum().sort_values(ascending = False) != 0]
data_type = final_df.dtypes.sort_values(ascending = False)[final_df.isnull().sum().sort_values(ascending = False) != 0]
Missing_val_matrix2 =pd.concat([total, data_type], axis=1, keys=['Total','Type']) 

"""Lets get dummy variable""""
final_features = pd.get_dummies(final_df).reset_index(drop=True) 
final_features.head()

final_features.drop(['Id'], axis = 1, inplace = True)               # drop id coloum from final_feature dataframe

"""*********************Feature Engineering******************************"""


##########Spliting of data to select the best model##############
X_train = final_features[:1460]

X_test = final_features[1460:]

y_train = pd.DataFrame(data =train["SalePrice"])                                     #needs to be a shape apperantly, same number of rows

######### for Principal component Analysis in machine learning#############
 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()                                                          #this is a requirement for pca .. otherwise it breaks 
X = sc.fit_transform(X_train)   

## splitting training and test(We are doing this to find the best model for our regression)
X_train_t,X_test_t,y_train_t,y_test_t = train_test_split(X_train,y_train,test_size =0.2, random_state = 1)
X_train_t
X_test_t
y_train_t.head()
y_test_t.head()
"""First try Multiple linear regression and calculate R square value"""

## Training the Multiple linear regression on the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_t,y_train_t)

#predicting the test set result

y_pred_t = regressor.predict(X_test_t)

##### Evaluating model performance##########
from sklearn.metrics import r2_score
r2_score(y_test_t,y_pred_t)               #-109.73346757356435

#### lets try svr
from sklearn.svm import SVR
regressor  = SVR(kernel = "rbf")
regressor.fit(X_train_t,y_train_t)
y_pred_t = regressor.predict(X_test_t)
r2_score(y_test_t,y_pred_t)              #-0.01643817383148427


#lets try decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train_t,y_train_t)
y_pred_t = regressor.predict(X_test_t)
r2_score(y_test_t,y_pred_t)              #0.6244180963904005


#LETS TRY random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100,random_state =0)
regressor.fit(X_train_t,y_train_t)
y_pred_t = regressor.predict(X_test_t)
r2_score(y_test_t,y_pred_t)               #0.8734636117347876



#********* Random forest regression medel is best suitable for this dataset as the value of R square is highest in that.#

"""Lets try random forest regression model on train and test set to predict final results"""

regressor = RandomForestRegressor(n_estimators = 100,random_state =0)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

   
y_pred = pd.DataFrame(y_pred,columns=["SalePrice"])
z = pd.DataFrame(test["Id"])
final_submission = pd.concat([z,y_pred],axis =1)    


final_submission.to_csv("MyFirstProject.csv",index= False)
final_submission








    
    