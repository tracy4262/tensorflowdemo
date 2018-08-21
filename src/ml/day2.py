"""
 * Created with PyCharm.
 * User: 彭诗杰
 * Date: 2018/8/20
 * Time: 9:10
 * Description: Simple Linear Regression
"""

# Step 1: Data Preprocessing

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[:, : 1].values
Y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 4, random_state=0)

# Step 2: Fitting Simple Linear Regression Model to the training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# Step 3: Predecting the Result

Y_pred = regressor.predict(X_test)

# Step 4: Visualization
# Visualising the Training results

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

# Visualizing the test results

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.show()
