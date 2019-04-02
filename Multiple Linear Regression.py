#Multiple Linear Regression Model 
#We will on modelling how R&D, Administration and Marketing Spending and the state will
#influence the profit of a company. There are 50 startups data in dataset.

#1 Importing the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

#2 Importing the dataset: 
dataset = pd.read_csv("50_Startups.csv")
#Y: dependent variable vector
#In the first run X's type is object due to the different types of independent variables. 
#State column contains categorical variables
X= dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 4].values

#3 Encoding the categorical variables: 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() #Change the text into numbers 0,1,2
X[: ,3]= labelencoder_X.fit_transform(X[: ,3])
onehotencoder= OneHotEncoder(categorical_features=[3])
#turn the numbers to dummy variables. Each column represents one state
#Compare the X and dataset tables to understand the relationship between the state and the columns 
X= onehotencoder.fit_transform(X).toarray()

#4 Avoid the dummy variables trap
#Delete the first column represent the California 
X= X[:, 1:]

#5 Splitting the dataset into the Training and Test dataset
#train_set_split: Split arrays or matrices into random train and test subsets
##random_state değeri sonuçların her seferinde aynı çıkmasını sağlamak için kullanılıyor.
#%20 of the dataset to the test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=0)
 
#6 Fit multiple Linear Regression model to our Train set
from sklearn.linear_model import LinearRegression
#Create an object called regressor in the LinearRegression class...
regressor = LinearRegression()
#Fit the linear regression model to the training set... We use the fit method
#the arguments of the fit method will be training sets 
regressor.fit(X_train,Y_train)

#7 Predicting the Test set results: 
y_pred= regressor.predict(X_test)


"""#Building the optimal Model using Backward Elimination""" 
#The backwards elimination function will give us the optimal variables from our data,
# Beta0 has x^0=1. Add a column of for the the first term of the MultiLinear Regression equation.
import statsmodels.formula.api  as sm 
#The 0th column contains only 1 in each 50 rows 
X= np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1) 
X_opt= X[:, [0,1,2,3,4,5]] #Optimal X contains the highly impacted independent variables
#OLS: Oridnary Least Square Class. endog is the dependent variable, exog is the number of observations
regressor_OLS=sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary() 

#constant for Beta0, x1 and x2 are the dummy variables for state, x3 is R&D,
#x4 is Administration, x5 is the marketing spends 
#Look at the highest p-values and remove it. In this condition x2(second 
#dummy variable has the highest one (0,990)

X_opt= X[:, [0,1,3,4,5]] 
regressor_OLS=sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary() 
#Run the three lines code and Look at the highest p-value again. First
#dummy variable, x1's p-value is 0,940. Remove this one

X_opt= X[:, [0,3,4,5]] 
regressor_OLS=sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary() 
#Run the three lines code again and Look at the highest p-value again. Admin
#spends (x2) has the highest p-value (0,602). Remove this one. 

X_opt= X[:, [0,3,5]] 
regressor_OLS=sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary() 
#Run the three lines code again and Look at the highest p-value again. Admin
#spends (x2) has the highest p-value (0,06). It depends on the significance 
#level determined. If your significance level is 0.05, it needs to remove 
#this one

X_opt= X[:, [0,3]] 
regressor_OLS=sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary() 
#That's it. We can see the highest impact variable is R&D spending in profit
#of these startups... 












