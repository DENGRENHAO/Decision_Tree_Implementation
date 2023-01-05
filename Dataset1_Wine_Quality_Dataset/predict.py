import sys
import numpy as np
from train import DecisionTreeClassifier
sys.path.insert(0, '..')
from Preprocess import *

# This gets the DecisionTreeClassifier, predict and output prediction of another test dataset
if __name__ == '__main__':
    # Train DecisionTreeClassifier with Grid Search
    grid = {'max_depth': [3, 5, 7, 9],
        'min_samples': [4, 6, 8, 10]}
    classifier = DecisionTreeClassifier(max_depth=7, min_samples=8, grid=grid)

    # Train DecisionTreeClassifier without Grid Search
    # classifier = DecisionTreeClassifier(max_depth=7, min_samples=8)

    # Read test datasets from excel file
    X_test_df = read_file('./Dataset1_test/X_test.xlsx')
    X_test = np.array(X_test_df, dtype='object')

    # Get prediction of test dataset from classifier
    prediction = classifier.predict(X_test)

    # Output prediction to excel file
    prediction_df = pd.DataFrame(prediction, columns=['class'])
    prediction_df.to_excel('./Dataset1_test/y_test_prediction.xlsx', index=False)
