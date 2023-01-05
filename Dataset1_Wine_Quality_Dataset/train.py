import sys
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
sys.path.insert(0, '..')
from Preprocess import *
from DecisionTree import DecisionTree


# This trains and tests Decision Tree on 'Dataset1_train' datasets
def DecisionTreeClassifier(max_depth=7, min_samples=8, grid=None):
    # Read datasets from excel file
    X_train_df = read_file('./Dataset1_train/X_train.xlsx')
    Y_train_df = read_file('./Dataset1_train/y_train.xlsx')

    # Fill nan with mean value
    fillna(X_train_df)

    # Over sampling on dataset
    ros = RandomOverSampler(random_state=42)
    x, y = ros.fit_resample(X_train_df, Y_train_df)
    x = np.array(x, dtype='object')
    y = np.array(y, dtype='object')

    # Split dataset for training and validation
    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Do grid search if needed
    if grid is not None:
        classifier = DecisionTree()
        grid_search_result = classifier.grid_search(grid, X_train, Y_train, X_val, Y_val)  # it's better to be used if validation or test dataset is provided
        # grid_search_result = classifier.grid_search(grid, X_train, Y_train)  # it's used when validation or test dataset isn't available
        max_depth=grid_search_result['best_max_depth']
        min_samples=grid_search_result['best_min_samples']
        print(f"Best Max Depth from Grid Search: {grid_search_result['best_max_depth']}, Best Min Samples from Grid Search: {grid_search_result['best_min_samples']}")

    # Train DecisionTree Classifier with training dataset
    print("Training......  Please wait......")
    print(f"Max Depth: {max_depth}, Min Samples: {min_samples}")
    classifier = DecisionTree(max_depth=max_depth, min_samples=min_samples)
    classifier.fit(X=X_train, Y=Y_train)
    print("Succeed Training!")
    print("Your Decision Tree Architecture:")
    classifier.print_tree()
    prediction = classifier.predict(X_val)

    # Get metric results on validation dataset
    metric_results = classifier.get_metric_results(y_test=Y_val, prediction=prediction)
    print("Validation Metric results:")
    print(metric_results)
    classifier.report(y_test=Y_val, prediction=prediction)

    return classifier