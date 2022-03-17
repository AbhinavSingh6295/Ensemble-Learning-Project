# Ensemble-Learning-Project

## Contributions - 
####  Decision Tree Implementation - Daniel Roca & Abhinav Singh

####  Cyberbullying on Twitter - Abhimanyu Soni, Kavya Shirke & Ruy Roa

## 1. Decision Tree Implementation from scratch

Edit the input data in the main.py file and run the decision tree implementation.

- The tree can handle classification or regression tasks. In the main.py set task = 'classification' or task = 'regression' depending on the specific need of machine learning task.

````
# Choose the task of the tree (either classification or regression)
task = "classification"
# task = "regression"
````

- The input data can be either a numpy array or a pandas dataframe. The pre-requisite is that the label is the last column in the data.
  ````
  # ------------------------------------
  # Classification Test Data
  # -------------------------------------
  N = 2000
  X, y = make_moons(N, noise=0.2)
  y = np.reshape(y, (N, 1))
  data = np.append(X, y, axis=1)

  train_data = data[:1000, :]
  val_data = data[1000:1500, :]
  test_data = data[1500:, :]
  ````

- After creating the decision_tree object, set the hyperparameters - max_depth, min_samples and loss criterion (Entropy or Gini Impurity). Then, train the decision tree on train dataset:
  ````
  dt = DecisionTree(train_data, task, max_depth=3)
  
  # Train the decision tree with cross-entropy method
  dt.train()
  ````

  - The decision tree can be used for calculating the accuracy/rmse on a validation set
    ````
    acc = dt.evaluation(data_predict)
    print(acc)
    ````

  - Also, it can be used for predicting on a testing set without lablels
    ````
    data_predict = test_data
    predictions = dt.predict(data_predict, val='validation')
    ````

- In order to simplify the decision tree, post-pruning can be performed, using the validation dataset.
  ````
  dt.post_pruning(train_data=train_data, val_data=val_data, ml_task=task)
  dt.print()
  ````
  
- With graphviz installed in the local environment, it can be used for visualizing the output tree, using the draw_tree function.
  ````
  # Draw the resulting tree
  dt.draw_tree()
  ````
  
  ![](DecisionTree/tree_sample.PNG)
  



