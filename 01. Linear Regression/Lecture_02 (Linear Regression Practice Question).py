# ===============================================================================================================================
# Project on LINEAR REGRESSION

#  --> You just got some contract work with an E-commerce company based in New York City that sells "clothing online"
#  --> but they also have in-store style and clothing advice sessions.

# --> Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either
# --> on a mobile app or website for the clothes they want.

# --> The company is trying to decide whether to focus their efforts on their mobile app experience or their website.
# --> They've hired you on contract to help them figure it out! Let's get started!
# ===============================================================================================================================

#  FIRSTLY IMPORT THE LIBRARIES (01) ___________________________________________________________________________________
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  WE READ SOME INFO FROM THE TABLE (02) _______________________________________________________________________________

#   READING THE DATA FROM THE FILE
customers = pd.read_csv("./Ecommerce Customers")

# 1.) checking the head() of the file : Syntax is .head(x), x refers to the number of rows of the table by default its set to 5
print(customers.head(6))

# 2.) checking the info() of the file
print(customers.info())

# 3.) checking the describe() of the file
print(customers.describe())

# Exploratory Data Analysis (03) _______________________________________________________________________________________
#  ---> Let's explore the data by creating a number of plots.

# Use sea born to create a joint plot to compare the Time on Website and Yearly Amount Spent columns.
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

# More time on site, more money spent.
# sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)   # this is for time on web site VS yearly amount spent
# plt.show()


# sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)        # this is for time on app site VS yearly amount spent
# plt.show()


# Use joint plot to create a 2D hex bin plot comparing Time on App and Length of Membership
# sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
# plt.show()


#   Exploring types of relationships across the entire data set using PAIR PLOT (04) ___________________________________
# sns.pairplot(customers)
# plt.show()

# ======================================================================================================================
# --> Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?
#       * Length of Membership
# ======================================================================================================================

#   CREATE A LINEAR MODEL PLOT FOR 'Yearly Amount Spent vs. Length of Membership' (05)__________________________________
# sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
# plt.show()

#   TRAINING AND TESTING DATA (06)______________________________________________________________________________________
# Now that we've explored the data a bit, let's go ahead and split the data into ''training'' and ''testing'' sets.

# ======================================================================================================================
# The "training data" is used to make sure the machine recognizes patterns in the data ----
# The "test data" is used to see how well the machine can ""predict"" new answers based on its training -----
# ======================================================================================================================

# Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column
y = customers['Yearly Amount Spent']                                                             # INDEPENDENT VARIABLE
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]  # DEPENDENT VARIABLE


# Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101
# Random State uses random permutations to generate the splits
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# TRAINING THE MODEL (07)_______________________________________________________________________________________________
# Now its time to train our model on our training data!
from sklearn.linear_model import LinearRegression

#  Create an instance of a LinearRegression() model named lm.
lm = LinearRegression()

# FIT lm ON THE TRAINING DATA (08) _____________________________________________________________________________________
lm.fit(X_train,y_train)

# PRINT OUT THE COEFFICIENTS OF THE MODEL (09) _________________________________________________________________________
# The coefficients
print('Coefficients: \n', lm.coef_)

# PREDICTING TEST DATA (10) ____________________________________________________________________________________________
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# Use lm.predict() to predict off the X_test set of the data
predictions = lm.predict( X_test)

# Create a scatter plot of the real test values versus the predicted values.
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

# EVALUATING THE MODEL (11) ____________________________________________________________________________________________
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2)
# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.

# calculate these metrics by hand! ================================
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# RESIDUALS (12)  ____________________________________________________________________________________________________
# You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data.

# Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().
sns.distplot((y_test-predictions),bins=50);


# CONCLUSION (13)   ____________________________________________________________________________________________________
# We still want to figure out the answer to the original question, do we focus our effort on mobile app or website development?
# Or maybe that doesn't even really matter, and Membership Time is what is really important. Let's see if we can interpret the
# coefficients at all to get an idea.

coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# Interpreting the coefficients:
#
# Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.



# Do you think the company should focus more on their mobile app or on their website?

# This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app,
# or develop the app more since that is what is working better. This sort of answer really depends on the other factors going on
# at the company, you would probably want to explore the relationship between Length of Membership and the App or the Website before
# coming to a conclusion!
