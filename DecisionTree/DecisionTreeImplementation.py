# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:21:31 2022

@author: Abhinav Singh & Daniel Roca
"""

# Imports
import numpy as np
from sklearn.datasets import make_moons

# Auxiliary functions
# Defines if a split has only one label or not
def is_split_pure(data):
    pure = True
    y_label = data[:, -1]
    count_labels = len(np.unique(y_label))

    if count_labels > 1:
        pure = False

    return pure

# This is the main function of the decision tree algorithm
def decision_tree(data, count=0, min_samples=2, max_depth=5):

    # If the data is pure?
    if is_split_pure(data):
        print("pure")


    else:
        print("not pure")

    # 1. Determine the best possible split
    # 2. Split the data

    # 3. Record the subtree
    # 4. Recursively run the algorithm in the subtrees

# -------------------------------------
# Classification Test Data
# -------------------------------------
N = 1000
X,y = make_moons(N, noise = 0.2)
y = np.reshape(y, (N, 1))
data = np.append(X, y, axis=1)
# -------------------------------------
# Regression Test Data
# -------------------------------------
# N = 200
# X = np.linspace(-1,1,N)
# Y = X**2 + np.random.normal(0,0.07,N)
# X = X.reshape(-1, 1)

# Calls to the main function
decision_tree(data)