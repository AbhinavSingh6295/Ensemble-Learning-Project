# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:21:31 2022

@author: Abhinav Singh & Daniel Roca
"""

# Imports
import numpy as np
import pandas as pd

# Auxiliary functions
# Preprocesses the data
def preprocess_data(d, cat_cols = None):
    # If the data is a numpy_array, convert to pandas dataframe
    if type(d).__module__ == np.__name__:
        d = pd.DataFrame(data=d)

    # TODO: would this be a good place to convert to dummies?
    # cat_cols = input list of categorical columns in data
    if cat_cols != None:
        d = pd.get_dummies(d, columns=cat_cols)

    # TODO: also I was thinking on naming the columns standard way [0, 1, ..., n] - We can do that becuase we don't keep the names in the decision tree process anyways
    d.columns = range(d.shape[1])

    return d


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
    entropy_value = np.sum(probs * -np.log2(probs))
    return entropy_value


# Split the data into left and right branch
def gini_impurity(d):
    y = d.iloc[:, -1]  # label column
    probs = y.value_counts() / y.shape[0]  # Probability of each label
    gini = 1 - np.sum(probs ** 2)
    return gini


# Updated, because previous method was not working in case column name are not indexes
def split(d, split_variable, split_value):
    data_left = d[d.iloc[:, split_variable] <= split_value]
    data_right = d[d.iloc[:, split_variable] > split_value]
    return data_left, data_right


# Finding all possible splits in data and choosing the one with the lowest entropy
# TODO - Should we give the hyperparameter to choose if we want to use gini or entropy for impurity calculation.
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
            # TODO: when do we use entropy and when do we use gini? -
            # TODO: They are more or less same, on internet it is mentioned that gini is
            # TODO: less computationally expensive than entropy. So maybe we can use gini, but not sure.
            proportion_left = data_left.shape[0] / (data_left.shape[0] + data_right.shape[0])
            proportion_right = data_right.shape[0] / (data_left.shape[0] + data_right.shape[0])
            ent = proportion_left * entropy(data_left) + proportion_right * entropy(data_right)

            if ent <= min_entropy:
                min_entropy = ent
                split_column = spColumn
                split_value = spValue

    return split_column, split_value


class DecisionTree:

    def __init__(self, train_data, ml_task, max_depth=None, min_samples=None):
        self.data = preprocess_data(train_data)
        self.ml_task = ml_task
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    # This is the main function of the decision tree algorithm
    # Prerequisites: data is a pandas dataframe with
    def decision_tree(self, d, ml_task, max_depth, min_samples, count=0):
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
            condition = "attribute*" + str(best_column) + " <= " + str(best_value)
            result_tree = {condition: []}

            # 4. Recursively run the algorithm in the subtrees
            result_tree[condition].append(
                self.decision_tree(data_left, ml_task, max_depth, min_samples, count=count))
            result_tree[condition].append(
                self.decision_tree(data_right, ml_task, max_depth, min_samples, count=count))
            return result_tree

    # When this function is called the attribute tree is calculated with the
    # decision tree function
    def train(self):
        self.tree = self.decision_tree(self.data, self.ml_task, self.max_depth, self.min_samples)
        print("Training complete!")
        print("Resulting tree: ")
        print(self.tree)

    # For prediction on one example
    # I changed it a bit.... it takes as input the whole dataset
    def predict(self, d):

        # It applies the same preprocessing steps for the prediction data
        test_data = preprocess_data(d)

        # TODO - Don't know if we need this condition becuase the test data can contain the label column as well, which can helps in calculating the accuracy.
        # If (test_columns + 1) != train_columns -> It cannot predict
        if test_data.shape[1] + 1 != self.data.shape[1]:
            print("The input dataset should have the same columns to use the tree for predictions")
            return None
        # If the tree is not trained yet
        elif self.tree is None:
            print("Use train function before predicting the data")
            return None
        else:
            predictions = []

            for index, row in test_data.iterrows():
                reach_leaf = False
                cut_tree = self.tree
                # TODO: this set of prints are used for understanding
                # TODO: how it is doing the prediction, we could comment them out
                print("values: ", row.values)

                # Evaluates iteratively the tree
                # If it finds another condition it keeps iterating i.e. going to lower
                # levels of the tree.
                # When it finds a leaf, it adds the value to predictions
                # and stops iterating.
                while not reach_leaf:
                    print("starts iter")
                    print(cut_tree)

                    condition = list(cut_tree.keys())[0]  # Node with condition for split
                    feature_index, operator, value = condition.split(" ")
                    column_name = feature_index.split("*")[1]

                    if row[int(column_name)] <= float(value):
                        # Goes to left node
                        if isinstance(cut_tree[condition][0], dict):
                            print("left sub-tree")
                            cut_tree = cut_tree.get(list(cut_tree.keys())[0])[0]
                        else:
                            print("left leaf")
                            predictions.append(cut_tree[condition][0])
                            reach_leaf = True
                    else:
                        # Goes to right node
                        if isinstance(cut_tree[condition][1], dict):
                            print("right sub-tree")
                            cut_tree = cut_tree.get(list(cut_tree.keys())[0])[1]
                        else:
                            print("right leaf")
                            predictions.append(cut_tree[condition][1])
                            reach_leaf = True

            print("Prediction process successful!")
            return predictions

    def accuracy(self, d):
        # Call the predict function
        predictions = self.predict(d)
        # Check for the rows where predictions are same as test labels
        correct_results = d.iloc[:, -1] == predictions
        return correct_results.mean()
