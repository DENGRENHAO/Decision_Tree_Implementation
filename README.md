# Classification Tree Implementation

## Overview
Build and train a decision tree with python by scratch. Tested on two kaggle datasets: 
1. [Wine Quality Dataset](https://www.kaggle.com/datasets/rajyellow46/wine-quality)
2. [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)

## Prerequisite

```
git clone https://github.com/DENGRENHAO/Decision_Tree_Implementation.git
pip install -r requirements.txt
```

## Code Explanation

### `DecisionTree.py`
For Decision Tree
- class `Node`: A class for all nodes in the decision tree.
- class `DecisionTree`: A class for getting an instance of the whole decision tree.
    - def `gini_index`: Calculate gini index for a dataset
    - def `split_dataset`: Split dataset into left and right for left and right children of a node
    - def `get_best_split`: Get the best split among all features and values with the best gini index. In this function, it calculates gini index incrementally, so it runs in linear time, not O(n^2). It speeds up the training process a lot.
    - def `build_tree`: Build decision tree
    - def `fit`: Train the model
    - def `predict`: Get all predict results for the whole dataset
    - def `record_prediction`: Get predict result for a single record
    - def `print_tree`: Print decision tree architecture
    - def `get_metric_results`: Return metric results for test datasets
    - def `report`: Print prediction metrics
    - def `grid_search`: Do grid search for multiple hyperparameters

### `Preprocess.py`

For preprocessing
- def `read_file`: Read file from excel file
- def `fillna`: Fill nan with mean value (only available on numerical datasets)
- def `removeNAN`: Remove row with nan values
- def `preprocess_data`: Preprocess text data
- def `tokenizer_preprocess`: Tokenize text data

### In Directory `Dataset1_Wine_Quality_Dataset`

- `train.py`
    - def `DecisionTreeClassifier`: This trains and tests Decision Tree on `Dataset1_train` datasets. The procedures are listed below:
        1. Read datasets from excel file `X_train.xlsx`, `Y_train.xlsx` (Code below shows that it reads and trains the training dataset from those TA provided)
            ```
            X_train_df = read_file('./Dataset1_train/X_train.xlsx')
            Y_train_df = read_file('./Dataset1_train/y_train.xlsx')
            ```
        2. Fill nan with mean value
        3. Over sampling on dataset
        4. Split dataset for training and validation (80% and 20% respectively)
        5. Do grid search if needed
        6. Train DecisionTree Classifier with training dataset
        7. Get metric results on validation dataset

- `predict.py`
    - In `main`: his gets the DecisionTreeClassifier, predict and output prediction of another test dataset. The procedures are listed below:
        1. Train DecisionTreeClassifier with or without Grid Search
        2. Read test datasets from excel file `X_test.xlsx` (Code below shows that it reads and test the test dataset from those TA provided.)
            ```
            X_test_df = read_file('./Dataset1_test/X_test.xlsx')
            ```
        3. Get prediction of test dataset from classifier
        4. Output prediction to excel file (Code below shows that it outputs the prediction results of the test dataset provided by TA. It outputs to the same folder `Dataset1_test` where `X_test.xlsx` exists.)
        ```
        prediction_df.to_excel('./Dataset1_test/y_test_prediction.xlsx', index=False)
        ```

### In Directory `Dataset2_Sentiment_Analysis_on_Movie_Reviews`

- `train.py`
    - def `tokenizeDataframe`: Tokenize dataset from pandas DataFrame, split dataset for training and validation, and convert to numpy array. The procedures are listed below:
        1. Under sampling on dataset
        2. Remove row with nan values
        3. Preprocess text data
        4. Split dataset for training and validation
        5. Tokenize text data
        6. Convert to numpy array
    - def `TfidfVectorizeDataframe`: TfidfVectorize dataset from pandas DataFrame, split dataset for training and validation, and convert to numpy array.
        1. Setup stemmer for English language
        2. Create stopword for Engilish language
        3. Create vectorizer
        4. Remove row with nan values
        5. Fit training dataset in vectorizer and transform training and validation dataset
        6. Split dataset for training and validation
        7. Turn sparse matrix to numpy array
        8. Use PCA to reduce feature number to increase training speed and avoid overfitting. Fit training dataset in pca and transform training and validation dataset
    - def `DecisionTreeClassifier`: This trains and tests Decision Tree on `Dataset2_train` datasets. The procedures are listed below:
        1. Read datasets from excel file `X_train.xlsx`, `Y_train.xlsx` (Code below shows that it reads and trains the training dataset from those TA provided)
            ```
            X_df = read_file('./Dataset2_train/X_train.xlsx')
            Y_df = read_file('./Dataset2_train/y_train.xlsx')
            ```
        2. Preprocess datasets
        3. Do grid search if needed
        4. Train DecisionTree Classifier with training dataset
        5. Get metric results on validation dataset

- `predict.py`
    - In `main`: his gets the DecisionTreeClassifier, predict and output prediction of another test dataset. The procedures are listed below:
        1. Read test datasets from excel file `X_test.xlsx` (Code below shows that it reads and test the test dataset from those TA provided.)
            ```
            X_test_df = read_file('./Dataset2_test/X_test.xlsx')
            ```
        2. Train DecisionTreeClassifier with or without Grid Search and and preprocess test datasets
        3. Get prediction of test dataset from classifier
        4. Output prediction to excel file (Code below shows that it outputs the prediction results of the test dataset provided by TA. It outputs to the same folder `Dataset2_test` where `X_test.xlsx` exists.)
        ```
        if preprocess_method.casefold() == 'Tokenize'.casefold():
            prediction_df.to_excel('./Dataset2_test/y_test_prediction_Tokenize.xlsx', index=False)
        else:
            prediction_df.to_excel('./Dataset2_test/y_test_prediction_Tfidf.xlsx', index=False)
        ```
    
## Usage

### Get prediction of `X_test` on Dataset1
```
cd /your/path/to/project/Dataset1_Wine_Quality_Dataset/
python -u "predict.py"
```

### Get prediction of `X_test` on Dataset2
```
cd /your/path/to/project/Dataset2_Sentiment_Analysis_on_Movie_Reviews/
```
Then, choose which preprocessing method you want. You can choose them by change two line of codes in `./Dataset2_Sentiment_Analysis_on_Movie_Reviews/predict.py`. There are two options: `Tokenize` and `TfidfVectorize`. Default is `TfidfVectorize`.
- Choose `Tokenize`: Uncomment `preprocess_method = 'Tokenize'` and comment `preprocess_method = 'TfidfVectorize'`.
- Choose `TfidfVectorize`: Uncomment `preprocess_method = 'TfidfVectorize'` and comment `preprocess_method = 'Tokenize'`.

After choosing preprocessing method:
```
python -u "predict.py"
```

## Evaluation

### Results of Dataset1 validation set

- Hyperparameters:
    - max_depth: 7
    - min_samples: 10
- Metrics
    - Accuracy: 0.6881
    - F1_score: 0.6909
    - Precision: 0.7094
    - Recall: 0.6878
- Confusion Matrix:

    |     | 3   | 4   | 5   | 6   | 7   | 8   |
    | --- | --- | --- | --- | --- | --- | --- |
    | 3   | 85  | 0   | 8   | 0   | 0   | 0   |
    | 4   | 0   | 52  | 10  | 14  | 2   | 0   |
    | 5   | 1   | 18  | 42  | 22  | 11  | 0   |
    | 6   | 0   | 4   | 14  | 47  | 13  | 1   |
    | 7   | 0   | 1   | 2   | 14  | 64  | 1   |
    | 8   | 0   | 0   | 0   | 9   | 15  | 63  |

- Classification Report:

    |     | Precision | Recall | F1-score |
    | --- | --------- | ------ | -------- |
    | 3   | 0.99      | 0.91   | 0.95     |
    | 4   | 0.69      | 0.67   | 0.68     |
    | 5   | 0.55      | 0.45   | 0.49     |
    | 6   | 0.44      | 0.59   | 0.51     |
    | 7   | 0.61      | 0.78   | 0.68     |
    | 8   | 0.97      | 0.72   | 0.83     |


### Results of Dataset2 validation set

- Preprocess with `Tokenize`
    - Hyperparameters:
        - max_depth: 5
        - min_samples: 6
    - Metrics
        - Accuracy: 0.3006
        - F1_score: 0.2890
        - Precision: 0.2912
        - Recall: 0.3006
    - Confusion Matrix:

        |     | 0   | 1   | 2   | 3   | 4   |
        | --- | --- | --- | --- | --- | --- |
        | 0   | 497 | 142 | 126 | 195 | 172 |
        | 1   | 301 | 157 | 282 | 253 | 138 |
        | 2   | 95  | 103 | 530 | 291 | 113 |
        | 3   | 262 | 133 | 264 | 302 | 171 |
        | 4   | 436 | 113 | 119 | 248 | 215 |

    - Classification Report:

        |     | Precision | Recall | F1-score |
        | --- | --------- | ------ | -------- |
        | 0   | 0.31      | 0.44   | 0.37     |
        | 1   | 0.24      | 0.14   | 0.18     |
        | 2   | 0.40      | 0.47   | 0.43     |
        | 3   | 0.23      | 0.27   | 0.25     |
        | 4   | 0.27      | 0.19   | 0.22     |

- Preprocess with `TfidfVectorize`
    - Hyperparameters:
        - max_depth: 9
        - min_samples: 10
    - Metrics
        - Accuracy: 0.5124
        - F1_score: 0.2401
        - Precision: 0.3011
        - Recall: 0.2517
    - Confusion Matrix:

        |     | 0   | 1   | 2     | 3    | 4   |
        | --- | --- | --- | ----- | ---- | --- |
        | 0   | 35  | 168 | 676   | 236  | 17  |
        | 1   | 65  | 388 | 3292  | 576  | 43  |
        | 2   | 90  | 493 | 11089 | 999  | 62  |
        | 3   | 54  | 351 | 3546  | 1233 | 84  |
        | 4   | 20  | 111 | 806   | 486  | 50  | 

    - Classification Report:

        |     | Precision | Recall | F1-score |
        | --- | --------- | ------ | -------- |
        | 0   | 0.13      | 0.03   | 0.05     |
        | 1   | 0.26      | 0.09   | 0.13     |
        | 2   | 0.57      | 0.87   | 0.69     |
        | 3   | 0.35      | 0.23   | 0.28     |
        | 4   | 0.20      | 0.03   | 0.06     |

## Discussion

### Wine Quality Dataset

In this dataset, it contains 1023 records and 11 features. All of the features are numerical. The number of records is not too big, so I do over sampling on dataset. Because some of the value are missed, I fill nan with mean value of that feature. 

Due to the small number of records and features, it only takes a few second to train the Decision Tree. Therefore, using grid search to find the best hyperparameter of `max_depth` and `min_samples` with brute force is feasible.

In the metric results of validation set, its performance is quite reasonable because this decision tree's architecture is a little simple. To improve the performance, using bagging algorithm like Random Forest or Boosting algorithm like Gradient Boosting and XGBoost is better for this problem.

### Sentiment Analysis on Movie Reviews

In this dataset, it contains 124848 records and a single feature. The aim of this dataset is to classify the sentiments of phrases taken from movie reviews in the Rotten Tomatoes dataset. Each phrase can be classified as one of the following 5 sentiments: negative, somewhat negative, neutral, somewhat positive, or positive.

Because this decision tree can only accept numerical features, it is necessary to convert the phrases to numerical features. Therefore, I provide two different methods to preprocess text data, which are `Tokenize` and `TfidfVectorize`. Following is the brief explanation.
- `Tokenize`: It uses standard text preprocessing method for NLP. The procedures are listed below.
    1. Remove HTML tags
    2. Remove words that don't start with English characters
    3. Tokenize words
    4. Remove stop words
    5. Lemmatize words
    
- `TfidfVectorize`: It uses TfidfVectorizer from Scikit-Learn package, which converts text to a matrix of TF-IDF features. TF-IDF is an abbreviation for Term Frequency Inverse Document Frequency. It utilizes Term Frequency and Inverse Document Frequency to convert text to a vector with numbers. The procedures are listed below.
    1. Stem words with SnowballStemmer from nltk package
    2. Tokenize words
    3. Remove stop words
    4. Vectorize text data using TfidfVectorizer
    5. Use PCA to reduce feature number to increase training speed and avoid overfitting.

- Comparison between `Tokenize` and `TfidfVectorize`:
    - `Tokenize`:
        - Advantages:
            - Only convert a phrase to 30 features of numbers, so the run time of training decision tree is way lower (About one minute for one epoch).
            - Due to the lower run time, it is feasible and recommended to use grid search to find the best hyperparameter of `max_depth` and `min_samples`.
        - Disadvantages:
            - The performance of the decision tree isn't good enough because the small number of features may cause the tree underfit on the dataset.
    - `TfidfVectorize`:
        - Advantages:
            - It not only focuses on the frequency of words present in the corpus but also provides the importance of the words.
            - The performance of the decision tree is better because the number of features is more.
        - Disadvantages:
            - Convert a phrase to 500 features of numbers, so the run time of training decision tree is way higher (About ten minute for one epoch).
            - Due to the higher run time, it is not feasible and not recommended to use grid search to find the best hyperparameter of `max_depth` and `min_samples`.

However, using these two preprocessing method with Decision Tree still can't have a good performance because decision tree's architecture is too simple. For the same as in Dataset1, to improve the performance, using bagging algorithm like Random Forest or Boosting algorithm like Gradient Boosting and XGBoost is better for this problem. Even so, these algorithm still can't get the best performance for this problem. To have the best performance for this problem, using transformers might be better.
