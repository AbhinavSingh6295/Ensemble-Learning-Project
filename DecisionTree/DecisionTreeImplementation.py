# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:21:31 2022

@author: Abhinav Singh & Daniel Roca
"""

# Imports
import numpy as np
import pandas as pd
import pydot
from PIL import Image
from io import BytesIO


# Auxiliary functions
# Preprocesses the data
def preprocess_data(d, cat_cols=None):
    # If the data is a numpy_array, convert to pandas dataframe
    if type(d).__module__ == np.__name__:
        d = pd.DataFrame(data=d)

    # cat_cols = input list of categorical columns in data
    if cat_cols is not None:
        d = pd.get_dummies(d, columns=cat_cols)

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
def best_split(d, criterion):
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

            proportion_left = data_left.shape[0] / (data_left.shape[0] + data_right.shape[0])
            proportion_right = data_right.shape[0] / (data_left.shape[0] + data_right.shape[0])

            if criterion == 'entropy':
                ent = proportion_left * entropy(data_left) + proportion_right * entropy(data_right)
            else:
                ent = proportion_left * gini_impurity(data_left) + proportion_right * gini_impurity(data_right)

            if ent <= min_entropy:
                min_entropy = ent
                split_column = spColumn
                split_value = spValue

    return split_column, split_value


class DecisionTree:

    def __init__(self, train_data, ml_task, max_depth=None, min_samples=None, criterion='entropy'):
        self.data = preprocess_data(train_data)
        self.ml_task = ml_task
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.criterion = criterion
        self.tree = None
        self.graph = None
        self.list_node_names = []

    # This is the main function of the decision tree algorithm
    # Prerequisites: data is a pandas dataframe with
    def decision_tree(self, d, ml_task, max_depth, min_samples, criterion, count=0):
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
            best_column, best_value = best_split(d, criterion=criterion)

            # 2. Split the data
            data_left, data_right = split(d, best_column, best_value)

            # 3. Record the subtree
            condition = "attribute*" + str(best_column) + " <= " + str(best_value)
            result_tree = {condition: []}

            # 4. Recursively run the algorithm in the subtrees
            result_tree[condition].append(
                self.decision_tree(data_left, ml_task, max_depth, min_samples, criterion, count=count))
            result_tree[condition].append(
                self.decision_tree(data_right, ml_task, max_depth, min_samples, criterion, count=count))
            return result_tree

    # When this function is called the attribute tree is calculated with the
    # decision tree function
    def train(self):
        self.tree = self.decision_tree(self.data, self.ml_task, self.max_depth, self.min_samples, self.criterion)
        print("Training complete!")
        print("Resulting tree: ")
        print(self.tree)

    def predict(self, d, val='validation'):
        # It applies the same preprocessing steps for the prediction data
        test_data = preprocess_data(d)

        if val == 'test' and test_data.shape[1] + 1 != self.data.shape[1]:
            print("The input dataset should have the same columns to use the tree for predictions and no labels")
            return None
        elif val == 'validation' and test_data.shape[1] != self.data.shape[1]:
            print("The input dataset should have the same columns to use the tree for predictions and the labels")
        # If the tree is not trained yet
        elif self.tree is None:
            print("Use train function before predicting the data")
            return None
        else:
            predictions = []

            for index, row in test_data.iterrows():
                reach_leaf = False
                cut_tree = self.tree
                # print("values: ", row.values)

                # Evaluates iteratively the tree
                # If it finds another condition it keeps iterating i.e. going to lower
                # levels of the tree.
                # When it finds a leaf, it adds the value to predictions
                # and stops iterating.
                while not reach_leaf:
                    # print("starts iter")
                    # print(cut_tree)

                    condition = list(cut_tree.keys())[0]  # Node with condition for split
                    feature_index, operator, value = condition.split(" ")
                    column_name = feature_index.split("*")[1]

                    if row[int(column_name)] <= float(value):
                        # Goes to left node
                        if isinstance(cut_tree[condition][0], dict):
                            # print("left sub-tree")
                            cut_tree = cut_tree.get(list(cut_tree.keys())[0])[0]
                        else:
                            # print("left leaf")
                            predictions.append(cut_tree[condition][0])
                            reach_leaf = True
                    else:
                        # Goes to right node
                        if isinstance(cut_tree[condition][1], dict):
                            # print("right sub-tree")
                            cut_tree = cut_tree.get(list(cut_tree.keys())[0])[1]
                        else:
                            # print("right leaf")
                            predictions.append(cut_tree[condition][1])
                            reach_leaf = True

            # print("Prediction process successful!")
            return predictions

    def evaluation(self, d):
        # Call the predict function
        predictions = self.predict(d, val="validation")

        if predictions is None:
            return None
        else:
            if self.ml_task == 'classification':
                correct_results = d[:, -1] == predictions
                accuracy = correct_results.mean()
                return 'Accuracy: {}'.format(accuracy)
            else:
                rmse = (((predictions - d[:, -1])**2).mean())**0.5
                return 'RMSE: {}'.format(rmse)

    def post_pruning(self, tree, train_data, val_data, ml_task):

        if tree == None:
            tree = self.tree

        # Pre-process train and val data
        train_data = preprocess_data(train_data)
        val_data = preprocess_data(val_data)

        # First node in the input tree
        question = list(tree.keys())[0]
        left_tree, right_tree = tree[question]

        # Base case - when both left and right answer is leaf
        if not isinstance(left_tree, dict) and not isinstance(right_tree, dict):

            # Prediction on val_data using original tree
            predictions = self.predict(val_data, val='validation')

            if ml_task.lower() == "classification":
                # Output, if there was no split
                leaf = train_data.iloc[:, -1].value_counts().index[0]
                # Error using the above output i.e, without split
                errors_leaf = sum(leaf != val_data.iloc[:, -1])
                # Error with existing split
                errors_decision_node = sum(predictions != val_data.iloc[:, -1])

            else:  # regression
                leaf = train_data.iloc[:, -1].mean()
                errors_leaf = ((leaf - val_data.iloc[:, -1]) ** 2).mean()
                errors_decision_node = ((predictions - val_data.iloc[:, -1]) ** 2).mean()

            if errors_leaf <= errors_decision_node:
                return leaf
            else:
                self.tree = tree  # replace the original tree with updated tree
                return tree

        else:  # When either of the answer are dictionary

            feature_idx, comparison, value = question.split()
            feature = feature_idx.split("*")[1]

            # train and validation datasets for the splits
            train_data_yes, train_data_no = train_data[train_data[int(feature)] <= float(value)], train_data[train_data[int(feature)] > float(value)]
            val_data_yes, val_data_no = val_data[val_data[int(feature)] <= float(value)], val_data[val_data[int(feature)] > float(value)]

            # Recursively calling the function
            if isinstance(left_tree, dict):
                left_tree = self.post_pruning(left_tree, train_data_yes, val_data_yes, ml_task)

            if isinstance(right_tree, dict):
                right_tree = self.post_pruning(right_tree, train_data_no, val_data_no, ml_task)

            tree = {question: [left_tree, right_tree]}

            self.tree = tree

            return tree

    # Auxiliary function for drawing recursively the tree
    def draw(self, parent_name, child_name):

        elem_exists = True
        while elem_exists:
            if child_name in self.list_node_names:
                child_name = child_name + " "
            else:
                elem_exists = False

        self.list_node_names.append(child_name)

        edge = pydot.Edge(parent_name, child_name)
        self.graph.add_edge(edge)

    # Auxiliary function for drawing recursively the tree
    def visit(self, node, parent=None):

        if isinstance(node, np.floating):
            self.draw(parent, str(node))
        else:
            for k, v in node.items():
                if isinstance(v, list):
                    if parent:
                        self.draw(parent, k)
                    self.visit(v[0], k)
                    self.visit(v[1], k)
                else:
                    self.draw(parent, k)
                    self.draw(k, k + '_' + v)

    def draw_tree(self):
        try:
            self.graph = pydot.Dot(graph_type='graph')
            self.visit(self.tree)
            Image.open(BytesIO(self.graph.create_png())).show()
        except Exception as e:
            print("Consider installing graphviz in your project environment")
            print("And adding to the path the Dot.exe route to be able to see the graph.")
            print("Original trace of the error:")
            print(e)

    def print(self):
        return print(self.tree)
