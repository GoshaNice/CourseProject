from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('data/result_225.csv')
X = dataset[['in_sigma_m', 'in_sigma_w', 'in_b', 'in_d']]
y = dataset['alpha'] # для alpha
# y = dataset['beta'] - для beta

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn import metrics

print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Absolute Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
