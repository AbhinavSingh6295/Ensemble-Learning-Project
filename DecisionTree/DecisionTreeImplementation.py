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
    # Updated, because previous method was not working in case column name are not indexes
    data_left = d[d.iloc[:, split_variable] <= split_value]
    data_right = d[d.iloc[:, split_variable] > split_value]
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
            # TODO: when do we use entropy and when do we use gini? - They are more or less same, on internet it is mentioned that gini is
            # less computationally expensive than entropy. So maybe we can use gini, but not sure.
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
def decision_tree(d, ml_task, count=0, max_depth=None, min_samples=None):
    # Check for max_depth condition
    if max_depth is None:
        depth_cond = True
    else:
        if count < max_depth:
            depth_cond = True
        else:
            depth_cond = False

    # Check for min_samples condition
    if min_samples is None:
        sample_cond = True
    else:
        if d.shape[0] < min_samples:
            sample_cond = True
        else:
            sample_cond = False

    # If the data is pure? or max depth / min sample condition reached
    if is_split_pure(d) or not depth_cond or not sample_cond:
        # Prediction from leaf node in case of classification
        if ml_task.lower() == "classification":
            classes, class_counts = np.unique(d.iloc[:, -1], return_counts=True)
            prediction = classes[class_counts.argmax()]
        # Prediction from leaf node in case of regression
        else:
            prediction = np.mean(d.iloc[:, -1])

        return prediction

    else:
        count += 1
        # 1. Determine the best possible split
        best_column, best_value = best_split(d)

        # 2. Split the data
        data_left, data_right = split(d, best_column, best_value)

        # 3. Record the subtree
        condition = "attribute_" + str(best_column) + " <= " + str(best_value)
        result_tree = {condition: []}

        # 4. Recursively run the algorithm in the subtrees
        result_tree[condition].append(
            decision_tree(data_left, ml_task, count=count, max_depth=max_depth, min_samples=min_samples))
        result_tree[condition].append(
            decision_tree(data_right, ml_task, count=count, max_depth=max_depth, min_samples=min_samples))
        return result_tree

### Prediction on test  ######
# For prediction on one example
def predict(row, tree):
    condition = list(tree.keys())[0] # Node with condition for split
    feature_index, operator, value = condition.split(" ")

    # Example in left node
    if row[int(feature_index)] <= float(value):
        # Use function recursively in case it's not a leaf node
        if isintance(tree[condition][0], dict):
            return predict(row, tree[condition][0])
        # Else return the value of node
        else:
            return tree[condition][0]

    # Example in right node
    else:
        if isinstance(tree[condition][1], dict):
            return predict(row, tree[condition][1])
        else:
            return tree[condition][1]

# For prediction of data in dataframe (df) #TODO: can come-up with better way to do this
predictions = df.apply(predict, args=(tree, ), axis=1)



# -------------------------------------
# Classification Test Data
# -------------------------------------
#N = 1000
#X, y = make_moons(N, noise=0.2)
#y = np.reshape(y, (N, 1))
#data = np.append(X, y, axis=1)
# -------------------------------------
# Regression Test Data
# -------------------------------------
N = 200
X = np.linspace(-1, 1, N)
y = X**2 + np.random.normal(0, 0.07, N)
data = np.array((X, y)).T

# Calls to the main function

# If the data is a numpy_array, convert to pandas dataframe
if type(data).__module__ == np.__name__:
    data = pd.DataFrame(data=data)

# Run the decision tree algorithm
print("Input data: ")
print(data.head(5))
#tree = decision_tree(data, 'classification', max_depth=3)
tree = decision_tree(data, 'regression', max_depth=3)
print(tree)
