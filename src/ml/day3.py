"""
 * Created with PyCharm.
 * User: 彭诗杰
 * Date: 2018/8/22
 * Time: 9:15
 * Description: Multiple linear regression
"""
# Step 1: Data Preprocessing
# Importing the libraries

import pandas as pd

# Importing the dataset

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding Categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy Variable Trap

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Step 2: Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Step 3: Predicting the Test set results

y_pred = regressor.predict(X_test)

print(y_pred)
