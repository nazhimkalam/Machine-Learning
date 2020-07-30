# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# READ A CSV FILE
df = pd.read_csv('kyphosis.csv')
print(df.head())

# PAIRPLOT
sns.pairplot(df,hue='Kyphosis')
plt.show()

# TRAIN TEST DATA
from sklearn.model_selection import train_test_split
X = df.drop('Kyphosis', axis=1) # independent variable
y = df['Kyphosis']              # dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# PREDICTIONS
predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

# RANDOM FOREST
from sklearn.ensemble import  RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfc_pred))

print(classification_report(y_test, rfc_pred))