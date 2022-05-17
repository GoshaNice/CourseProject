from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import pandas as pd
import numpy as np

dataset = pd.read_csv('data/result_225.csv')
X = dataset[['in_sigma_m', 'in_sigma_w', 'in_b', 'in_d']]
y = dataset['alpha'] # для alpha
# y = dataset['beta'] - для beta


for k in range(2, 5):
    poly_reg = PolynomialFeatures(degree=k)
    X_poly = poly_reg.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    print('---------------')
    print(f'degree : {k}')
    print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Absolute Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
