from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

dataset_gosha = pd.read_csv('data/result_Gosha.csv')
dataset_david = pd.read_csv('data/result_David.csv')
dataset_ksenia = pd.read_csv('data/result_Ksenia.csv')
dataset_danila = pd.read_csv('data/result_Danila.csv')

dataset = dataset_gosha.append(dataset_david)
dataset = dataset.append(dataset_ksenia)
dataset = dataset.append(dataset_danila)

X = dataset[['in_sigma_m', 'in_sigma_w']]
y = dataset['alpha'] # - для alpha
#y = dataset['beta'] - для beta

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn import metrics

print(f'coefficients : {regressor.coef_}')
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Absolute Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
