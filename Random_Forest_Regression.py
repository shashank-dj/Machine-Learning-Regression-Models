#Random Forest Regression:

#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2 Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set

#3 Fitting the Random Forest Regression Model to the dataset
# Create RF regressor here
from sklearn.ensemble import RandomForestRegressor 
#Put 10 for the n_estimators argument. n_estimators mean the number of trees in the forest.
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,y)

#4 Predicting a new result
y_pred = regressor.predict(5.5)


#5 Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Check It (Random Forest Regression Model)w/100 trees')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
