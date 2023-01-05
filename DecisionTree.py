import numpy as np
from tqdm.contrib import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# fFor all nodes in the decision tree
class Node():
    def __init__(self, feature_index=None, value=None, left=None, right=None, leaf_value=None):
        self.feature_index = feature_index
        self.value = value
        self.left = left
        self.right = right
        self.leaf_value = leaf_value

# For getting an instance of the whole decision tree
class DecisionTree():
    def __init__(self, max_depth=8, min_samples=10):
        self.root = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        
    # Calculate gini index for a dataset
    def gini_index(self, dataset):
        values, counts = np.unique(dataset[:, -1] , return_counts=True)
        size = dataset.shape[0]
        gini = 1.0
        for val, cnt in zip(values, counts):
            gini -= (cnt / size)**2
        return gini
    
    # Split dataset into left and right for left and right children of a node
    def split_dataset(self, dataset, index, value):
        left_dataset = np.array([row for row in dataset if row[index] <= value])
        right_dataset = np.array([row for row in dataset if row[index] > value])
        return left_dataset, right_dataset
    
    # Get the best split among all features and values with the best gini index
    # In this function, it calculates gini index incrementally, so it runs in linear time, not O(n^2).
    # Speeds up the training process a lot.
    def get_best_split(self, dataset):
        best_split = {}
        best_split['gini'] = 1.0
        for index in range(dataset.shape[1] - 1):
            dataset = dataset[dataset[:, index].argsort()]  # sort dataset by 'index' column
            unique_values, cnts = np.unique(dataset[1:, -1], return_counts=True)
            left_dict = {}
            left_dict[dataset[0, index]] = 1
            right_dict = dict(zip(unique_values, cnts))
            left_gini, right_gini = 0.0, 1.0
            for val, cnt in zip(unique_values, cnts):
                right_gini -= (cnt / (dataset.shape[0] - 1))**2
            for i in range(1, dataset.shape[0] - 1):
                left_prev_size = i
                right_prev_size = dataset.shape[0] - left_prev_size
                left_cur_size = i + 1
                right_cur_size = dataset.shape[0] - left_cur_size
                cur_value = dataset[i, -1]
                if cur_value not in left_dict:
                    left_dict[cur_value] = 1
                    left_gini = (left_cur_size**2 - (left_prev_size**2 * (1 - left_gini) + 1)) / left_cur_size**2
                else:
                    left_gini = (left_cur_size**2 - (left_prev_size**2 * (1 - left_gini) - left_dict[cur_value]**2 + (left_dict[cur_value] + 1)**2)) / left_cur_size**2
                    left_dict[cur_value] += 1
                if right_dict[cur_value] == 1:
                    right_dict[cur_value] = 0
                    right_gini = (right_cur_size**2 - (right_prev_size**2 * (1 - right_gini) - 1)) / right_cur_size**2
                else:
                    right_gini = (right_cur_size**2 - (right_prev_size**2 * (1 - right_gini) - right_dict[cur_value]**2 + (right_dict[cur_value] - 1)**2)) / right_cur_size**2
                    right_dict[cur_value] -= 1
                gini = left_gini * left_cur_size / (float)(dataset.shape[0]) + right_gini * right_cur_size / (float)(dataset.shape[0])
                if gini < best_split['gini']:
                    best_split['gini'] = gini
                    best_split['index'] = index
                    best_split['value'] = dataset[i, index]
                    best_split['left_dataset'] = dataset[:i, :]
                    best_split['right_dataset'] = dataset[i:, :]
                        
        return best_split if best_split['gini'] < 1 else None
    
    # Build decision tree
    def build_tree(self, dataset, cur_depth=0):
        num_samples = dataset.shape[0]
        if cur_depth < self.max_depth and num_samples >= self.min_samples:
            best_split = self.get_best_split(dataset)
            if best_split is not None:
                left = self.build_tree(best_split['left_dataset'], cur_depth + 1)
                right = self.build_tree(best_split['right_dataset'], cur_depth + 1)
                return Node(best_split['index'], best_split['value'], left, right)
        
        values, counts = np.unique(dataset[:, -1] , return_counts=True)
        return Node(leaf_value=values[counts.argmax()])
            
    # Train the model
    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = np.expand_dims(Y, axis=1)
        self.root = self.build_tree(np.concatenate((X, Y), axis=1))
        
    # Get all predict results for the whole dataset
    def predict(self, dataset):
        return np.array([self.record_prediction(record, self.root) for record in dataset])
        
    # Get predict result for a single record
    def record_prediction(self, record, node):
        if node.leaf_value is not None:
            return node.leaf_value
        if record[node.feature_index] <= node.value:
            return self.record_prediction(record, node.left)
        else:
            return self.record_prediction(record, node.right)
        
    # Print decision tree architecture
    def print_tree(self, node=None, cur_depth=0):
        if node is None:
            node = self.root
        if node.leaf_value is not None:
            print("|   "*cur_depth + "|--- Leaf Node Value: " + str(node.leaf_value))
        else:
            print("|   "*cur_depth + "|--- Split Feature Index: " + str(node.feature_index) + ",  Split Value: " + str(node.value))
            print("|   "*cur_depth + "|--- Left Node:")
            self.print_tree(node.left, cur_depth + 1)
            print("|   "*cur_depth + "|--- Right Node:")
            self.print_tree(node.right, cur_depth + 1)
            
    # Return metric results for test datasets
    def get_metric_results(self, y_test, prediction):
        Y_test = y_test.tolist()
        metric_results = {}
        metric_results['accuracy'] = accuracy_score(Y_test, prediction)
        metric_results['f1_score'] = f1_score(Y_test, prediction, average="macro")
        metric_results['precision'] = precision_score(Y_test, prediction, average="macro")
        metric_results['recall'] = recall_score(Y_test, prediction, average="macro")
        return metric_results

    # Print prediction metrics
    def report(self, prediction, y_test):
        Y_test = y_test.tolist()
        print('Confusion Matrix:')
        print(confusion_matrix(Y_test, prediction))
        print('Classification Report:')
        print(classification_report(Y_test, prediction))
            
    # Do grid search for multiple hyperparameters
    def grid_search(self, grid, X_train, Y_train, X_test=None, Y_test=None):
        max_depth_list = grid['max_depth']
        min_samples_list = grid['min_samples']
        
        if not max_depth_list:
            max_depth_list.append(8)
        if not min_samples_list:
            min_samples_list.append(10)
        
        if X_test is None or Y_test is None:
            X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

        summary = {}
        summary['best_f1_score'] = 0
        summary['history'] = []
        for max_depth, min_samples in itertools.product(max_depth_list, min_samples_list):                
            classifier = DecisionTree(max_depth, min_samples)
            classifier.fit(X=X_train, Y=Y_train)
            prediction = classifier.predict(X_test)
            metric_results = classifier.get_metric_results(y_test=Y_test, prediction=prediction)
            history = {}
            history['max_depth'] = max_depth
            history['min_samples'] = min_samples
            history['accuracy'] = metric_results['accuracy']
            history['f1_score'] = metric_results['f1_score']
            history['precision'] = metric_results['precision']
            history['recall'] = metric_results['recall']
            summary['history'].append(history)
            print("max_depth: " + str(max_depth) + ", min_samples: " + str(min_samples) + ", f1_score: " + str(metric_results['f1_score']))

            if metric_results['f1_score'] > summary['best_f1_score']:
                summary['best_f1_score'] = metric_results['f1_score']
                summary['best_max_depth'] = max_depth
                summary['best_min_samples'] = min_samples
        return summary