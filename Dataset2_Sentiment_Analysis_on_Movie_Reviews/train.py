import sys
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, '..')
from Preprocess import *
from DecisionTree import DecisionTree

# Tokenize dataset from pandas DataFrame, split dataset for training and validation, and convert to numpy array
def tokenizeDataframe(X_df, Y_df):
    # Under sampling on dataset
    rus = RandomUnderSampler(random_state=42)
    X_resampled_df, Y_resampled_df = rus.fit_resample(X_df, Y_df)

    # Remove row with nan values
    removeNAN(X_resampled_df, Y_resampled_df)

    # Preprocess text data
    X_text = preprocess_data(X_resampled_df)

    # Split dataset for training and validation
    Y_dataset = Y_resampled_df.Sentiment.values
    X_train, X_val, Y_train, Y_val = train_test_split(X_text, Y_dataset, test_size=0.2, stratify=Y_dataset, random_state=42)

    # Tokenize text data
    X_train, X_val, tokenizer, len_max = tokenizer_preprocess(X_train, X_val)

    # Convert to numpy array
    X_train = np.array(X_train, dtype='object')
    X_val = np.array(X_val, dtype='object')

    return X_train, X_val, Y_train, Y_val, tokenizer, len_max


# TfidfVectorize dataset from pandas DataFrame, split dataset for training and validation, and convert to numpy array
def TfidfVectorizeDataframe(X_df, Y_df):
    # Setup stemmer for English language
    stemmer = SnowballStemmer(language='english')

    # Function to create tokenizer
    def tokenize(text):
        return [stemmer.stem(token) for token in word_tokenize(text)]
    
    # Create stopword for Engilish language
    eng_stopword = stopwords.words('english')

    # Create vectorizer
    vectorizer = TfidfVectorizer(tokenizer=tokenize,
                                stop_words = eng_stopword,
                                ngram_range=(1,2),
                                max_features=500)

    # Remove row with nan values       
    removeNAN(X_df, Y_df)

    # Fit training dataset in vectorizer and transform training and validation dataset
    vectorizer.fit(X_df.Phrase)
    X_dataset = vectorizer.transform(X_df.Phrase)
    Y_dataset = Y_df.Sentiment.values

    # Split dataset for training and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X_dataset, Y_dataset, test_size=0.2, stratify=Y_dataset, random_state=42)

    # Turn sparse matrix to numpy array
    X_train = X_train.toarray()
    X_val = X_val.toarray()

    # Use PCA to reduce feature number to increase training speed and avoid overfitting
    # Fit training dataset in pca and transform training and validation dataset
    pca = PCA(n_components=400)
    _ = pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)

    return X_train, X_val, Y_train, Y_val, vectorizer, pca

# This trains and tests Decision Tree on 'Dataset2_train' datasets
def DecisionTreeClassifier(max_depth=9, min_samples=10, preprocess_method='TfidfVectorize', train_size=1000, grid=None):
    # Read datasets from excel file
    X_df = read_file('./Dataset2_train/X_train.xlsx')
    Y_df = read_file('./Dataset2_train/y_train.xlsx')

    # Preprocess datasets
    if preprocess_method.casefold() == 'Tokenize'.casefold():
        X_train, X_val, Y_train, Y_val, tokenizer, len_max = tokenizeDataframe(X_df, Y_df)
    else:
        X_train, X_val, Y_train, Y_val, vectorizer, pca = TfidfVectorizeDataframe(X_df, Y_df)

    # Do grid search if needed
    if grid is not None:
        classifier = DecisionTree()
        grid_search_result = classifier.grid_search(grid, X_train[:train_size], Y_train[:train_size], X_val, Y_val)  # it's better to be used if validation or test dataset is provided
        # grid_search_result = classifier.grid_search(grid, X_train, Y_train)  # it's used when validation or test dataset isn't available
        max_depth=grid_search_result['best_max_depth']
        min_samples=grid_search_result['best_min_samples']
        print(f"Best Max Depth from Grid Search: {grid_search_result['best_max_depth']}, Best Min Samples from Grid Search: {grid_search_result['best_min_samples']}")

    # Train DecisionTree Classifier with training dataset
    print("Training......  Please wait......")
    print(f"Max Depth: {max_depth}, Min Samples: {min_samples}")
    classifier = DecisionTree(max_depth=max_depth, min_samples=min_samples)
    classifier.fit(X=X_train[:train_size], Y=Y_train[:train_size])
    print("Succeed Training!")
    print("Your Decision Tree Architecture:")
    classifier.print_tree()
    prediction = classifier.predict(X_val)

    # Get metric results on validation dataset
    metric_results = classifier.get_metric_results(y_test=Y_val, prediction=prediction)
    print("Validation Metric results:")
    print(metric_results)
    classifier.report(y_test=Y_val, prediction=prediction)

    if preprocess_method.casefold() == 'Tokenize'.casefold():
        return classifier, tokenizer, len_max
    else:
        return classifier, vectorizer, pca