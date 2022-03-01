# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:21:31 2022

@author: Abhinav Singh & Daniel Roca
"""

# Imports
import numpy as np
from sklearn.datasets import make_moons
import pandas as pd


# Auxiliary functions
# Defines if a split has only one label or not
def is_split_pure(d):
    pure = True
    y_label = d.iloc[:, -1]
    count_labels = len(np.unique(y_label))

    if count_labels > 1:
        pure = False

    return pure


# Determine the best split using Entropy and Gini Impurity
def entropy(d):
    y = d.iloc[:, -1]  # label column
    probs = y.value_counts() / y.shape[0]  # Probability of each label
    entropy = np.sum(probs * -np.log2(probs))
    return entropy


def gini_impurity(d):
    y = d.iloc[:, -1]  # label column
    probs = y.value_counts() / y.shape[0]  # Probability of each label
    gini = 1 - np.sum(probs ** 2)
    return gini


# Split the data into left and right branch
def split(d, split_variable, split_value):

    #TODO: Abhi - I was not able to do the datatype thing to
    #TODO: distinguish between continous of categorical.
    #TODO: I commented it out and we can improve it for next versions

    # For Categorical Column
    #if d[[split_variable]].dtypes == 'O':
    #    data_left = d[d[split_variable] == split_value]
    #    data_right = d[d[split_variable] != split_value]

    # For Continous Split Column
    #else:
    data_left = d[d[split_variable] <= split_value]
    data_right = d[d[split_variable] > split_value]

    return data_left, data_right


# Finding all possible splits in data and choosing the one with the lowest entropy
def best_split(d):
    possible_splits = {}
    for column_index in range(d.shape[1] - 1):
        values = d.iloc[:, column_index]
        unique_values = np.unique(values)
        possible_splits[column_index] = unique_values

    # Define initial variables to calculate the best split
    min_entropy = 10e7
    split_column = -1
    split_value = -1

    # Iterate over the df columns
    for spColumn in possible_splits:
        # Iterate over the df_column values
        for spValue in possible_splits[spColumn]:
            # Split the data based on the value
            data_left, data_right = split(d, spColumn, spValue)
            # TODO: when do we use entropy and when do we use gini?
            proportion_left = data_left.shape[0] / (data_left.shape[0] + data_right.shape[0])
            proportion_right = data_right.shape[0] / (data_left.shape[0] + data_right.shape[0])
            ent = proportion_left * entropy(data_left) + proportion_right * entropy(data_right)

            if ent <= min_entropy:
                min_entropy = ent
                split_column = spColumn
                split_value = spValue

    return split_column, split_value


# This is the main function of the decision tree algorithm
# Prerequisites: data is a pandas dataframe with
def decision_tree(d, count=0, min_samples=2, max_depth=5):
    # If the data is pure?
    if is_split_pure(d):
        # TODO: review if this code is correct after implementing the recursive function
        classes, class_counts = np.unique(d[:, -1], return_counts=True)
        return classes[class_counts.argmax()]

    else:
        count += 1
        print("not pure")
        # 1. Determine the best possible split
        best_column, best_value = best_split(d)
        print(best_column)
        print(best_value)

        # 2. Split the data



        # 3. Record the subtree
        # 4. Recursively run the algorithm in the subtrees


# -------------------------------------
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
# X = np.linspace(-1,1,N)
# Y = X**2 + np.random.normal(0,0.07,N)
# X = X.reshape(-1, 1)

# Calls to the main function

# If the data is a numpy_array, convert to pandas dataframe
if type(data).__module__ == np.__name__:
    data = pd.DataFrame(data=data)

# Run the decision tree algorithm
print("Input data: ")
print(data.head(5))
decision_tree(data)
