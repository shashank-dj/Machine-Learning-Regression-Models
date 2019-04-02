#SIMPLE LINEAR REGRESSION: 

#NumPy is the fundamental package for scientific computing with Python. It contains among other things:

#1 Libraries importes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
# Single selections using iloc and DataFrame
# Rows:
data.iloc[0] # first row of data frame (Aleshia Tomkiewicz) - Note a Series data type output.
data.iloc[1] # second row of data frame (Evan Zigomalas)
data.iloc[-1] # last row of data frame (Mi Richan)
# Columns:
data.iloc[:,0] # first column of data frame (first_name)
data.iloc[:,1] # second column of data frame (last_name)
data.iloc[:,-1] # last column of data frame (id)
''''
#2 Importing the dataset: 
#The iloc indexer for Pandas Dataframe is used for integer-location based indexing / selection by position.
#X is a matrix and matrix of features and matrix of independent variable
#Y is a vector and vector of the independent variable.
dataset = pd.read_csv("Salary_Data.csv")
X= dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 1].values

#3 Splitting the data into training sets and test sets: 
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 1/3, random_state=1)

#4 Fitting the Simple Linear Regression to the Training Set 
# In library from sklearn we import the class which is linear regression class: 
from sklearn.linear_model import LinearRegression
#Create an object called regressor... Ordinary least squares Linear Regression.
regressor = LinearRegression()
#Fit the linear regression model to the training set... We use the fit method
#the arguments of the fit method will be training sets 
regressor.fit(X_train,Y_train)

#5 Predicting the Test set Results: 
#Lets put all the predicted salaries into a single vector. 
# Use the prediction method for the some observations. 
#These predictions are applied to test data X and we will get the predicted Y values... 
Y_pred = regressor.predict(X_test)
#Compare the y_pred vs Y_test

#6 Visualizing the training set results: 
#import matplotlib.pyplot as plt (for plotting the graph)
plt.scatter(X_train, Y_train, color='Blue') #Real Values 
plt.plot(X_train, regressor.predict(X_train), color='Red' ) #Plot Regression Line(Predicted values)
plt.title('Salary vs Experience (Traininig set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#7 Visualizing the test set results: 
plt.scatter(X_test, Y_test, color='Blue') #Test set 
plt.plot(X_train, regressor.predict(X_train), color='Red' ) #The regressor trained on training set 
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
#The dot values are test values. The linear regression line willbe the same as the previous one. 














