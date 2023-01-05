import sys
import numpy as np
from keras.utils import pad_sequences
from train import DecisionTreeClassifier
sys.path.insert(0, '..')
from Preprocess import *

# This gets the DecisionTreeClassifier, predict and output prediction of another test dataset
if __name__ == '__main__':
    # Preprocess with 'Tokenize' method:
    preprocess_method = 'Tokenize'
    # Preprocess with 'TfidfVectorize' method:
    # preprocess_method = 'TfidfVectorize'

    # Train DecisionTreeClassifier with Grid Search
    # grid = {'max_depth': [5, 7, 9, 11],
    #     'min_samples': [6, 8, 10, 12]}
    # Train DecisionTreeClassifier without Grid Search
    grid = None
    
    # Read test datasets from excel file
    X_test_df = read_file('./Dataset2_test/X_test.xlsx')

    # Train DecisionTreeClassifier and preprocess test datasets
    # Max train_size: 20000
    train_size = 20000
    if preprocess_method.casefold() == 'Tokenize'.casefold():
        classifier, tokenizer, len_max = DecisionTreeClassifier(max_depth=5, min_samples=6, preprocess_method=preprocess_method, train_size=train_size, grid=grid)
        X_text = preprocess_data(X_test_df)
        X_test = tokenizer.texts_to_sequences(X_text)
        X_test = pad_sequences(X_test, maxlen=len_max)
    else:
        classifier, vectorizer, pca = DecisionTreeClassifier(max_depth=9, min_samples=10, preprocess_method=preprocess_method, train_size=train_size, grid=grid)
        X_test = vectorizer.transform(X_test_df.Phrase)
        X_test = X_test.toarray()
        X_test = pca.transform(X_test)

    # Get prediction of test dataset from classifier
    prediction = classifier.predict(X_test)

    # Output prediction to excel file
    prediction_df = pd.DataFrame(prediction, columns=['Sentiment'])

    if preprocess_method.casefold() == 'Tokenize'.casefold():
        prediction_df.to_excel('./Dataset2_test/y_test_prediction_Tokenize.xlsx', index=False)
    else:
        prediction_df.to_excel('./Dataset2_test/y_test_prediction_Tfidf.xlsx', index=False)
