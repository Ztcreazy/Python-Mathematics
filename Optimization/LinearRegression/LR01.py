import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

"""
Ordinary Least Squares
min_w ||X*w - y||_2 **2
y_hat(w,x) = w_0 + w_1 * x_1 + w_2 * x_2 + ... w_p * x_p
w = (w_1, w_2, ..., w_p) coefficient
w_0 intercept
The least squares solution is computed using the singular value decomposition of X. 
If X is a matrix of shape (n_samples, n_features) this method has a cost of
O(n_samples*n_features **2), assuming that n_samples >= n_features
"""

diabete_X, diabete_y = datasets.load_diabetes(return_X_y=True)
# print(diabete_X) print(diabete_y)
print(diabete_X.size)
print(diabete_y.size)

diabete_X = diabete_X[:, np.newaxis, 2]
# print(diabete_X)
print(diabete_X.size)

diabetes_X_train = diabete_X[:-20]
diabetes_X_test = diabete_X[-20:]

diabetes_y_train = diabete_y[:-20]
diabetes_y_test = diabete_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()