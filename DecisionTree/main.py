# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:21:31 2022

@author: Abhinav Singh & Daniel Roca
"""
from sklearn.datasets import make_moons
import numpy as np
from DecisionTreeImplementation import DecisionTree

# Choose the task of the tree (either classification or regression)
task = "classification"
# task = "regression"

# Choose the data, in this we can use either numpy array or pandas dataframe
# The decision tree takes the last column of the pandas dataframe as the label
# and all the other columns as predicting variables.

# ------------------------------------
# Classification Test Data
# -------------------------------------
N = 1000
X, y = make_moons(N, noise=0.2)
y = np.reshape(y, (N, 1))
data = np.append(X, y, axis=1)

# -------------------------------------
# Regression Test Data
# -------------------------------------
# N = 200
# X = np.linspace(-1, 1, N)
# y = X**2 + np.random.normal(0, 0.07, N)
# data = np.array((X, y)).T

dt = DecisionTree(data, task, max_depth=3)

# Train the decision tree with cross-entropy method
dt.train()

# After this, the dt has the trained tree. Now we can make predictions with data that
# has the same structure

data_predict = np.array([[2, 2, 1], [0, 0, 0]])
predictions = dt.predict(data_predict, val='validation')
print(predictions)
acc = dt.accuracy(data_predict)
print(acc)


# Draw the resulting tree
dt.draw_tree()


