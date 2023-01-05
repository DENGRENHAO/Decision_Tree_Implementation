import re
import warnings
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

# Read file from excel file
def read_file(path):
    rawdata = pd.read_excel(path, header=0)
    return rawdata

# Fill nan with mean value (only available on numerical datasets)
def fillna(df):
    for idx in range(df.shape[1]):
        mean_value = df.iloc[:, idx].mean()
        df.iloc[:, idx].fillna(value=mean_value, inplace=True)

# Remove row with nan values
def removeNAN(x_df, y_df):
    nan_idx = x_df[x_df['Phrase'].isnull()].index.tolist()
    print("NaN indices: " + str(nan_idx))
    x_df.drop(nan_idx, axis=0, inplace=True)
    y_df.drop(nan_idx, axis=0, inplace=True)

# Preprocess text data
def preprocess_data(df):
    reviews = []
    for raw in tqdm(df['Phrase']):
        warnings.filterwarnings('ignore', category=UserWarning, module='bs4')
        text = BeautifulSoup(raw, 'lxml').get_text()
        only_text = re.sub('[^a-zA-Z]', ' ', text)
        words = word_tokenize(only_text.lower())
        stops = set(stopwords.words('english'))
        non_stopwords = [word for word in words if not word in stops]
        lemmatizer = WordNetLemmatizer()
        lemma_words = [lemmatizer.lemmatize(word) for word in non_stopwords]    
        reviews.append(lemma_words)
    return reviews

# Tokenize text data
def tokenizer_preprocess(list_X_train, list_X_val):
    unique_words = set()
    len_max = 0
    for sent in tqdm(list_X_train):
        unique_words.update(sent)
        if len_max < len(sent):
            len_max = len(sent)

    tokenizer = Tokenizer(num_words=len(list(unique_words)))
    tokenizer.fit_on_texts(list(list_X_train))
     
    X_train = tokenizer.texts_to_sequences(list_X_train)
    X_train = pad_sequences(X_train, maxlen=len_max)

    X_val = tokenizer.texts_to_sequences(list_X_val)
    X_val = pad_sequences(X_val, maxlen=len_max)

    return X_train, X_val, tokenizer, len_max