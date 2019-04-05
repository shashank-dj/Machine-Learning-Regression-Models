import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures



dataset = pd.read_csv("2016dolaralis.csv")


X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

X=X.reshape(251,1)
y=y.reshape(251,1)

plt.scatter(X,y)
plt.show()

#Lineer Reg.
lineermodel = LinearRegression()
lineermodel.fit(X,y)
lineermodel.predict(X)

plt.plot(X,lineermodel.predict(X),c="blue")


#Polinom Reg.#2.derece

tahminpolinom = PolynomialFeatures(degree=3)
Xyeni = tahminpolinom.fit_transform(X)

polinommodel = LinearRegression()
polinommodel.fit(Xyeni,y)
polinommodel.predict(Xyeni)

plt.plot(X,polinommodel.predict(Xyeni),c="red")

#Polinom Reg.#3.derece

tahminpolinom = PolynomialFeatures(degree=8)
Xyeni = tahminpolinom.fit_transform(X)

polinommodel = LinearRegression()
polinommodel.fit(Xyeni,y)
polinommodel.predict(Xyeni)

plt.plot(X,polinommodel.predict(Xyeni),c="magenta")

plt.show()

hatakaresilineermodel = 0
hatakaresipolinommodel = 0

for i in range(len(Xyeni)):
    hatakaresipolinommodel = hatakaresipolinommodel + (float(y[i])-float(polinommodel.predict(Xyeni)[i]))**2

for i in range(len(y)):
    hatakaresilineermodel = hatakaresilineermodel + (float(y[i])-float(lineermodel.predict(x)[i]))**2




hatakaresipolinommodel = 0
    
for a in range(150):

    tahminpolinom = PolynomialFeatures(degree=a+1)
    Xyeni = tahminpolinom.fit_transform(X)

    polinommodel = LinearRegression()
    polinommodel.fit(Xyeni,y)
    polinommodel.predict(Xyeni)
    for i in range(len(Xyeni)):
        hatakaresipolinommodel = hatakaresipolinommodel + (float(y[i])-float(polinommodel.predict(Xyeni)[i]))**2
    print(a+1,"inci dereceden fonksiyonda hata,", hatakaresipolinommodel)  
    
    hatakaresipolinommodel = 0

#En düşük hata 8. dereceden polinomla elde ediliyor. 
