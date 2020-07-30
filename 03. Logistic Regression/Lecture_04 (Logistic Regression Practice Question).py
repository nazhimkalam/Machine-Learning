# Logistic Regression Practice Question

# ======================================================================================================================
# In this project we will be working with a fake advertising data set, indicating"""" whether or not"""" a particular internet
# user clicked on an Advertisement on a company website. We will try to create a model that will predict whether or not
# they will click on an ad based off the features of that use
# ======================================================================================================================

# IMPORTING LIBRARIES (01) _____________________________________________________________________________________________
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================================================================
# Get the Data (02) ____________________________________________________________________________________________________
# Read in the advertising.csv file and set it to a data frame called ad_data.
ad_data = pd.read_csv('../items/advertising.csv')
# checking out the head()
print(ad_data.head())
# checking out the columns of the table
print(ad_data.columns)
# use info and describe of the ad_data
print(ad_data.info())
print(ad_data.describe())

# ======================================================================================================================
# Exploratory Data Analysis (03) _______________________________________________________________________________________

# Let's use 'seaborn' to explore the data!
# Try recreating the plots shown below!

# Create histogram of the Age
sns.set_style('whitegrid')
ad_data['Age'].plot.hist(bins=40)
plt.xlabel('Age')

# Create a jointplot showing Area Income versus Age.
sns.jointplot(x='Age', y='Area Income', data=ad_data)

# Create a jointplot showing the 'kde' distributions of' Daily Time spent on site' vs. 'Age'.
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');

# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')

#  Create a pairplot with the hue defined by the 'Clicked on Ad' column feature
fig = sns.pairplot(data=ad_data,hue='Clicked on Ad',palette='bwr')
# fig.savefig('pairplotLecture_04.png', dpi=500)
plt.show()

# ======================================================================================================================
# Logistic Regression (04) _____________________________________________________________________________________________
# Now it's time to do a train test split, and train our model!
# You'll have the freedom here to choose columns that you want to train on!

# Split the data into 'training set' and 'testing set' using ''''train_test_split''''

from sklearn.model_selection import train_test_split

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# 'Train and fit' a logistic regression model on the 'training set'.
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# ======================================================================================================================
# Predictions and Evaluations (05) _____________________________________________________________________________________
# Now predict values for the 'testing data'
predictions = logmodel.predict(X_test)

# ======================================================================================================================
# Create a classification report for the model.(06) ____________________________________________________________________
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))



