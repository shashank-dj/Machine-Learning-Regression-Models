import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')

"""x=dataset.loc[:, :["YearsExperience"]].values
y=dataset.loc[:, ["Salary"]].values"""

X= dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 1].values

linearregresyon = lr()

linearregresyon.fit(X,Y)

linearregresyon.predict(X) 

print("Eğim: ", linearregresyon.coef_)
print("Kesişme: ", linearregresyon.intercept_)
m= linearregresyon.coef_
b= linearregresyon.intercept_

print("Denklem")
print("y= ", m,"*x+",b)

#0'dan 149'a kadar bir matrix oluşturduk
a=np.arange(15)

plt.scatter(X,Y)
plt.plot(a, m*a+b)
plt.show() 