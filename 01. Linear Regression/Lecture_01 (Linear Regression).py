# Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
USAhousing = pd.read_csv('./USA_Housing.csv')  # this is the data file
# print(USAhousing.head())                   # this we display the first row set of the data table
# print(USAhousing.info())                   # gives information about the table
# print(USAhousing.describe())               # describes the data with new attributes such as mean, count, max, min, std etc...
# print(USAhousing.columns)                    # returns  an array of column names

# EDA { Exploratory Data Analysis }
# diagram = sns.pairplot(USAhousing)
# diagram.savefig('usaHousing.pdf',dip=1500) # i saved it as png so its much clear to see the details

# sns.distplot(USAhousing['Price'])
# sns.heatmap(USAhousing.corr())
# plt.show()

# Training a Linear Regression Model
# X and y arrays
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

# Train Test Split
#    Now let's split the data into a 'training set' and a 'testing set'.
#    We will 'train out model on the training set' and 'then use the test set to evaluate the model.'
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
# print(' x-training-set')
# print(X_train.head())
# print(' x-testing-set')
# print(X_test.head())

# Creating and Training the Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

# Model Evaluation
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.
# print the intercept
# print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
# print(coeff_df)

# Interpreting the coefficients:
#
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Income** is associated with an **increase of \$21.52 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area House Age** is associated with an **increase of \$164883.28 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Rooms** is associated with an **increase of \$122368.67 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Bedrooms** is associated with an **increase of \$2233.80 **.
# - Holding all other features fixed, a 1 unit increase in **Area Population** is associated with an **increase of \$15.15 **.

# Predictions from our Model
predictions = lm.predict(X_test)
# plt.scatter(y_test,predictions)

# Residual Histogram
# sns.distplot((y_test-predictions),bins=50);
# plt.show()


# Regression Evaluation Metrics
# Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# Mean Squared Error** (MSE) is the mean of the squared errors:
# Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:

# Comparing these metrics:
#
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
