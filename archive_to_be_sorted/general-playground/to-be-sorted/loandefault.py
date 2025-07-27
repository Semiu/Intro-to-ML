# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:00:24 2019

@author: Semiu: Database Systems CSci 765 Project -Data Mining Model for Predicting Loan 
Default
Dataset can be downloaded from the supplementary material section of this journal article:
https://www.tandfonline.com/doi/full/10.1080/10691898.2018.1434342
"""
"""
Reading the United States Small Business Administration Dataset into a Dataframe

Importing the Pandas library and confirming the success of the dataset reading by showing 
the first five rows of the data.
"""
import pandas as pd
datadf = pd.read_csv("SBAnational.csv", low_memory=False)
datadf.head(5)
"""
Checking if there are missing values in the dataset.
"""
datadf.isnull().sum()
"""
With the existence of missing values of categorical data types, it is important to firstly 
handle categorical data types (ordinal, class labels and nominals) before treating the missing 
values.
a. Each of the hypothetically important features with Text datatype is checked to identify 
the nominal and ordinal types for their corresponding data handling
b. The unique identifier is set as the index
"""
datadf['ApprovalDate']
datadf2 = datadf.set_index("LoanNr_ChkDgt", drop = False)
"""
Based on (a) above, RevLineCr, LowDoc and MIS_Status (all categorical data) are handled
using the ordinal/nominal feature encoding methods
"""
class_mapping1 = {'Y': 1, 'N': 0}
datadf2['RevLineCr'] = datadf2['RevLineCr'].map(class_mapping1)
datadf2['RevLineCr']

class_mapping2 = {'Y': 1, 'N': 0}
datadf2['LowDoc'] = datadf2['LowDoc'].map(class_mapping2)
datadf2['LowDoc']

class_mapping3 = {'P I F': 1, 'CHGOFF': 0}
datadf2['MIS_Status'] = datadf2['MIS_Status'].map(class_mapping3)
datadf2['MIS_Status']
"""
The NaN in  RevLineCr, LowDoc and MIS_Status are replaced with 0
"""
datadf2['RevLineCr'].fillna(0, inplace=True)

datadf2['LowDoc'].fillna(0, inplace=True)

datadf2['MIS_Status'].fillna(0, inplace=True)

"""
The other missing data features are replaced with the most frequent value of their respective columns
"""
datadf2 = datadf2.fillna(datadf2.mode().iloc[0])
"""
Checking if there are still missing values in the dataset.
"""
datadf2.isnull().sum()
"""
Assign the target variable of the data frame to another variable; "datadf2_y"
"""
datadf2_y = datadf2['MIS_Status']
"""
Remove the target variable (MIS_Status) and other variables (LoanNr_ChkDgt, Name, City, State, Bank, Bank State, NAICS) that are intuitively 
insignifcant from  datadf2. The datadf2.columns[] is used to get the position of each column, after the last dropping, for its precise dropping. 
"""
#The target variabble MIS_Status
datadf2.drop(datadf2.columns[23], axis= 1, inplace=True)

#The variabble LoanNr_ChkDgt
datadf2.drop(datadf2.columns[0], axis= 1, inplace=True)

#The variabble Name
datadf2.drop(datadf2.columns[0], axis= 1, inplace=True)

#The variabble City
datadf2.drop(datadf2.columns[0], axis= 1, inplace=True)

#The variabble State
datadf2.drop(datadf2.columns[0], axis= 1, inplace=True)

#The variabble Zip
datadf2.drop(datadf2.columns[0], axis= 1, inplace=True)

#The variabble Bank
datadf2.drop(datadf2.columns[0], axis= 1, inplace=True)

#The variabble BankState
datadf2.drop(datadf2.columns[0], axis= 1, inplace=True)

#The variabble NAICS
datadf2.drop(datadf2.columns[0], axis= 1, inplace=True)

"""
The remaining columns (after all the droppings) are the predictors, thus assigned to another variable; datadf2_x
"""
datadf2_x = datadf2
datadf2_x
"""
The columns with currency and comma are now treated to be fit for model building
These are: DisbursementGross, BalanceGross, ChgOffPrinGr, GrApprv and SBA_Apprv  
"""
datadf2_x[datadf2_x.columns[13:]] = datadf2_x[datadf2_x.columns[13:]].apply(lambda x: x.str.replace('$','')).apply(lambda x: x.str.replace(',','')).astype(float)
"""
The datadf2_x['ChgOffDate'] is observed as misleading because its issing values ought not be filled.
It is then dropped. It is as column [11] at this stage of data cleansing.
"""
datadf2_x.drop(datadf2_x.columns[11], axis= 1, inplace=True)
"""
The columns with date format are ApprovalDate and DisbursementDate in form of 13-Mar-10.
These are coverted to their respective years. For example, 13-Mar-10 will become 2010.
"""
#Approval Date
datadf2_x['ApprovalDate'] = pd.to_datetime(datadf2_x['ApprovalDate'], format='%d-%b-%y').dt.strftime('%Y')
 
#DisbursementDate
datadf2_x['DisbursementDate'] = pd.to_datetime(datadf2_x['DisbursementDate'], format='%d-%b-%y').dt.strftime('%Y')

#ApprovalFY 
datadf2_x.drop(datadf2_x.columns[1], axis= 1, inplace=True) 
  
"""
The dataset is then split to the train and test set. The test size is set at 20%
"""
from sklearn.cross_validation import train_test_split
x, y = datadf2_x.values, datadf2_y.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

"""
Training the dataset with the basic Decision Tree Classifier
"""
from sklearn import tree
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(x_train, y_train)

"""
Testing the accuracy of the Decision Tree Classifier
"""
clf1_sc = clf1.score(x_test, y_test)
print(clf1_sc)
"""
Training the dataset with the Gradient Boosting Classifier
"""
from sklearn.ensemble import GradientBoostingClassifier
clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train, y_train)

"""
Testing the accuracy of the Gradient Boosting Classifier
"""
clf2_sc = clf2.score(x_test, y_test)
print(clf2_sc)

clf2.score(x_train, y_train)

"""
Comparing the Basic Decision Tree and Boosted Decision Tree with a Histogram
"""
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
# Creating an empty Dataframe with column names as 
dfCLF = pd.DataFrame(columns=['Clf', 'Model'])

# Append rows in Empty Dataframe by adding dictionaries
dfCLF = dfCLF.append({'Clf': clf1_sc, 'Model': 'Basic Decision Tree'}, ignore_index=True)
dfCLF = dfCLF.append({'Clf': clf2_sc, 'Model': 'Boosted Decision Tree'}, ignore_index=True)
sns.barplot(x='Model', y='Clf', data=dfCLF)

"""
Model Performance Evaluation 1: 
Receiver Operating Characteristic (ROC) Curve and 
Area Under Curve (AUC) for the Boosted Decision Tree 
"""
from sklearn.metrics import roc_curve, auc

plt.figure(figsize = (10, 6))
plt.plot([0,1], [0,1], 'r--')

probs = clf2.predict_proba(x_test)

# Reading probability of second class (Pay-In-Full)
probs = probs[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

label = 'Boosted Decision Tree Model AUC:' + ' {0:.2f}'.format(roc_auc)
plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 4)
plt.xlabel('False Positive Rate', fontsize = 14)
plt.ylabel('True Positive Rate', fontsize = 14)
plt.title('Receiver Operating Characteristic', fontsize = 14)
plt.legend(loc = 'lower right', fontsize = 14)

"""
Model Performance Evaluation 2: 
Cumulative Accuracy Profile (CAP) Curve for the Boosted Decision Tree 
"""
#total number of y_test data instances
total_y_test = len(y_test)
#total number of class 1.0 (PIF)
class_1_count = np.sum(y_test)

print (class_1_count) 
#Total number of class 0.0 (CHGOFF)
class_0_count = total_y_test - class_1_count

print (class_0_count) 

plt.figure(figsize = (10, 6))

#Random Model -suggesting that detection of class 1.0 will grow linearly.
plt.plot([0, total_y_test], [0, class_1_count], c = 'r', linestyle = '--', label = 'Random Model')

#Perfect Model detecting the class 1.0 data instances
plt.plot([0, class_1_count, total_y_test],[0, class_1_count, class_1_count], c = 'grey', linewidth = 2, label = 'Perfect Model')

#Boosted Decision Tree Model
model_y = [y for _, y in sorted(zip(probs, y_test), reverse = True)]
y_values = np.append([0], np.cumsum(model_y))
x_values = np.arange(0, total_y_test + 1)
plt.plot(x_values, y_values, c = 'b', label = 'Boosted Decision Tree', linewidth = 4)

plt.xlabel('Total Observations', fontsize = 14)
plt.ylabel('Class 1.0 Observations', fontsize = 14)
plt.title('Cumulative Accuracy Profile (CAP)', fontsize = 14)
plt.legend(loc = 'lower right', fontsize = 14)

"""
Model Performance Evaluation 3: 
Cumulative Accuracy Profile (CAP) Analysis Using AUC and Plot for the Boosted Decision Tree
(with class_1_observed) 
"""
# Area under Random Model
a = auc([0, total_y_test], [0, class_1_count])

# Area between Perfect and Random Model
aP = auc([0, class_1_count, total_y_test], [0, class_1_count, class_1_count]) - a

# Area between Trained and Random Model
aR = auc(x_values, y_values) - a

#Accuracy Rate for Boosted Decision Tree: 0.9959399421174234
print("Accuracy Rate for Boosted Decision Tree with class 1: {}".format(aR / aP))

#Using Plot with CAP
index = int((50*total_y_test / 100))

## 50% Verticcal line from x-axis
plt.plot([index, index], [0, y_values[index]], c ='g', linestyle = '--')

## Horizontal line to y-axis from prediction model
plt.plot([0, index], [y_values[index], y_values[index]], c = 'b', linestyle = '--')

class_1_observed = y_values[index] * 100 / max(y_values)

print('The percentage of class_1_observed is : ', class_1_observed)