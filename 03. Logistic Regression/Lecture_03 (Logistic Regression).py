# For this lecture we will be working with Titanic Data Set from kaggle.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_train.csv')
# print(train)
# print(train.head())
print(train.info())
# print(train.columns)
#
# # We can use seaborn to create a simple heatmap to see where we are missing data!
# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.show()

# sns.set_style('whitegrid')
# sns.countplot(x='Survived',data=train,palette='RdBu_r')
# plt.show()


# sns.set_style('whitegrid')
# sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
# plt.show()


# sns.set_style('whitegrid')
# sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
# plt.show()


# sns.set_style('whitegrid')
# sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
# plt.show()


# sns.set_style('whitegrid')
# train['Age'].hist(bins=30,color='green',alpha=0.5)
# plt.show()

# sns.set_style('whitegrid')
# sns.countplot(x='SibSp',data=train)
# plt.show()

# sns.set_style('whitegrid')
# train['Fare'].hist(color='green',bins=40,figsize=(8,5))  # (columns, rows)
# plt.show()

# Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age
# data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
# However we can be smarter about this and check the average age by passenger class. For example:

# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
# plt.show()

#  We can see the wealthier passengers in the higher classes tend to be older, which makes sense.
#  We'll use these average age values to impute based on Pclass for Age.
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

# Let's check the heat map again
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Great! Let's go ahead and drop the Cabin column and the row in Embarked that is NaN.
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.dropna(inplace=True)

# #
# Converting Categorical Features
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm
# won't be able to directly take in those features as inputs

train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()


# Building a Logistic Regression model
# Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).
#
# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),
                                                    train['Survived'], test_size=0.30,
                                                    random_state=101)

# Training and Predicting
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# Evaluation

# We can check precision,recall,f1-score using classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))