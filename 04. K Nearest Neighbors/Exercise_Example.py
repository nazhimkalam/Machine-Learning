# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# READING DATA FROM A FILE
# Get the data by reading the KNN_Project_Data.csv file into a dataframe
df = pd.read_csv('KNN_Project_Data')
print(df.head())
print(df.columns)

# CREATING A PAIR PLOT
# Using pairplot to display all the possible outcomes
# sns.pairplot(data=df, hue='TARGET CLASS',palette='coolwarm')

# STANDARDIZING THE VARIABLES
#   * this is because the values in the csv file have big difference in the range EG:- 150 and 3000, therefore we have to standardize

# Import StandardScaler from Scikit learn
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler() object called scalar
scalar = StandardScaler()

# Fit scalar to the features
scalar.fit(df.drop('TARGET CLASS', axis=1))

# Use transform() to transform the features to a scaled version
scaled_features = scalar.transform(df.drop("TARGET CLASS",axis=1))

# Convert the scaled features to a dataframe
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat.head())

# TRAIN TEST SPLIT
#       we can split data to training set and testing set
from sklearn.model_selection import train_test_split
X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# USING KNN

# imports
from sklearn.neighbors import  KNeighborsClassifier

# create a KNN model with n_neighbors=1
knn = KNeighborsClassifier(n_neighbors=1)  # n_neighbors = 1 means k = 1

# Fit this KNN model to the training data
knn.fit(X_train, y_train)


# PREDICTIONS AND EVALUATIONS

# Use the predict method to predict values using your KNN model and X_test
pred = knn.predict(X_test)

# Create a confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))

# CHOOSING THE CORRECT K VALUE
# By choosing the correct K or n_neighbour value we can reduce the % error or increase the precision
# So how can we find the exact K value,  this can be achieved by drawing a graph
# By using a loop which calculates the error for a range of K values and plotting a graph
error_rate = []

for i in range (1,60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,60), error_rate, color='blue', linestyle='--', marker = 'o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs K')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

#  58 seems to look good
knn = KNeighborsClassifier(n_neighbors=58)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report (y_test, pred))
