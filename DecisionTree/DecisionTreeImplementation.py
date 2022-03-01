# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:21:31 2022

@author: Abhinav Singh & Daniel Roca
"""

# Imports
import numpy as np
from sklearn.datasets import make
import pandas as pd

# Auxiliary functions
# Defines if a split has only one label or not
def is_split_pure(data):
    pure = True
    y_label = data[:, -1]
    count_labels = len(np.unique(y_label))

    if count_labels > 1:
        pure = False

    return pure

# Determine the best split using Entropy and Gini Impurity
def entropy(data):
    y = data[:, -1] # label column
    probs = y.value_counts() / y.shape[0] # Probability of each label
    entropy = np.sum(probs * -np.log2(probs))
    return entropy

def gini_impurity(data):
    y = data[:, -1] # label column
    probs = y.value_counts() / y.shape[0] # Probability of each label
    gini = 1 - np.sum(probs ** 2)
    return gini

# Finding all possible splits in data
def possible_splits(data):
    possible_splits = {}
    for column_index in range(data.shape[1] - 1):
        values = data[:, column_index]
        unique_values = np.unique(values)
        possible_splits[column_index] = unique_values
    return possible_splits

# Split the data into left and right branch
def split(data, split_variable, split_value):

    # For Categorical Column
    if split_column_values.dtypes == 'O':
        data_left = data[data[split_variable] == split_value]
        data_right = data[data[split_variable] != split_value]

    # For Continous Split Column
    else:
        data_left = data[data[split_variable] <= split_value]
        data_right = data[data[split_variable] > split_value]

    return data_left, data_right

#




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