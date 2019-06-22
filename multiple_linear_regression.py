import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
"""
Features and Response: 
     
1.CRIM: per capita crime rate by town 
2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
3. INDUS: proportion of non-retail business acres per town
4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
5. NOX: nitric oxides concentration (parts per 10 million) 
6. RM: average number of rooms per dwelling 
7. AGE: proportion of owner-occupied units built prior to 1940 
8. DIS: weighted distances to ﬁve Boston employment centers
9. RAD: index of accessibility to radial highways 
10. TAX: full-value property-tax rate per $10,000
11. PTRATIO: pupil-teacher ratio by town
12. B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 
13. LSTAT: % lower status of the population 
14. MEDV: Median value of owner-occupied homes in $1000s 
"""
column_names= ["CRIM", "ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO"
               ,"B","LSTAT", "MEDV"]
#Load the dataset
boston_dataset= pd.read_csv("housing.csv",delim_whitespace=True, names=column_names)

#Correlation matrix to look at the correlation btw the variables: 
correlation_matrix= boston_dataset.corr()
f, ax= plt.subplots(figsize=(12,10))
sns.heatmap(correlation_matrix,vmax=1, square=True)
# Determine the independent (x) variables matrix and dependent (y) variable vector
# RM,PTRATIO,LSTAT are the top three correlated varaiables with MEDV  
X=boston_dataset.iloc[: ,[5,10,12]]
y=boston_dataset.iloc[:,13]
#Split the data into training and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.4, random_state=1)
#Create linear regression object
from sklearn.linear_model import LinearRegression
regressor=LinearRegression() 
#Fit multiple Linear Regression model to our Train set
regressor.fit(X_train,y_train)
# Predicting the Test set results: 
y_pred= regressor.predict(X_test)

# regression coefficients 
print('Coefficients: \n', regressor.coef_) 

# variance score: 1 means perfect prediction 
print("Variance score: {}".format(regressor.score(X_test,y_test)))


# plot for residual error 
## setting plot style 
plt.style.use("fivethirtyeight")
## plotting residual errors in training data 
plt.scatter(regressor.predict(X_train), regressor.predict(X_train)-y_train, color="green",
            marker="o",label="Train Data")
## plotting residual errors in test data 
plt.scatter(regressor.predict(X_test), regressor.predict(X_test)-y_test, color="red",
            marker="*",label="Test Data")
## plotting line for zero residual error 
plt.hlines(y=0,xmin=0, xmax=50,linewidth=2)
## plotting legend 
plt.legend(loc="upper right")
## plot title 
plt.title("Residual Errors")
## function to show plot 
plt.show()








