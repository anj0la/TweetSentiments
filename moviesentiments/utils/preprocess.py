"""
File: preprocess.py

Author: Anjola Aina
Date Modified: October 22nd, 2024

Description:

This file is used to preprocess data.
"""
import csv
import emoji
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def save_to_csv(cleaned_text: list[str], labels: list[str], file_path: str) -> None:
    """
    Saves the cleaned text and corresponding labels into a CSV file.
        
    Args:
    cleaned_text (list[str]: The cleaned text data.
    labels (list[str]): The labels for each piece of text.
    file_path (str): The path to save the CSV file.
    """
    fields = ['review', 'sentiment']
    rows = []
    for sentence, label in zip(cleaned_text, labels):
        rows.append({'review': sentence, 'sentiment': label})
    with open(file=file_path, mode='w', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        csv_file.close()  

def encode_labels(df: pd.DataFrame) -> list[str]:
    """
    Converts labels of each review into 0 (positive) or 1 (negative).

    Args:
        df (pd.DataFrame): The reviews Pandas DataFrame.

    Returns:
        list[str]: The list of labels.
    """
    if 'sentiment' not in df.columns:
        raise ValueError('Expected column "sentiment" in input file.')
    sentiments = df['sentiment']
    labels = []
    
    for sentiment in sentiments:
        labels.append(0) if sentiment == 'positive' else labels.append(1) # sentiment == 'negative'
            
    return labels

def clean_review(df: pd.DataFrame) -> np.ndarray:
    """
    Cleans a review.

    Args:
        df (pd.DataFrame): The dataframe containing the reviews to be cleaned.

    Raises:
        ValueError: Occurs if the column "review" does not exist in the dataframe.

    Returns:
        np.ndarray: The cleaned reviews.
    """
    if 'review' not in df.columns:
        raise ValueError('Expected column "review" in input file.')
    data = df['review']
    
    # Convert the text to lowercase
    data = data.str.lower()
    
    # Remove Unicode characters (non-ASCII)
    data = data.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
    
    # Remove punctuation, special characters, emails and links
    data = data.replace(r'[^\w\s]', '', regex=True)  # Removes non-alphanumeric characters except whitespace
    data = data.replace(r'http\S+|www\.\S+', '', regex=True)  # Remove URLs
    data = data.replace(r'\w+@\w+\.com', '', regex=True)  # Remove emails
    
    # Convert emojis to text
    data = data.apply(lambda x: emoji.demojize(x))
    data = data.replace(r':(.*?):', '', regex=True)
        
    # Remove stop words and apply lemmatization
    stop_words = set(stopwords.words('english'))
    data = data.apply(lambda sentence: ' '.join(WordNetLemmatizer().lemmatize(word) for word in sentence.split() if word not in stop_words))
    
    return data.values
            
def preprocess(file_path: str, output_file_path: str) -> None:
    """
    Preprocesses the text data, returning the processed data and corresponding labels.
     
    This function preprocesses the text data by converting the text to lowercase and emojis to text, removing punctuation, special characters,
    links, email addresses and applying lemmatization.
        
    Args:
        file_path (str): The file path containing the text data.
        output_file_path (str): The file path to put the cleaned text data into.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Clean text
    cleaned_text = clean_review(df)
    
    # Encode labels
    encoded_labels = encode_labels(df['sentiment'])
    
    # Save data to new CSV file
    save_to_csv(cleaned_text, encoded_labels, output_file_path)

def word_to_idx(text, vocab, unk_index):
    return [vocab.get(word, unk_index) for word in text.split()]

def build_vocab(text: pd.Series, min_token_length: int = 1) -> dict:
    """
    Tokenizes text and builds the vocabulary with the specified minimum token length. It is assumed that the text has been preprocessed.

    Args:
        text (list): The processed text containing all reviews.
        min_token_length (int): Minimum length of tokens to keep in the vocabulary. Default is 2.

    Returns:
        dict: The resulting vocabulary built from the unique tokens in the text.
    """
    tokenized_sentences = []
    all_tokens = []

    for sentence in text:
        # Tokenize and filter tokens based on min_token_length
        tokens = [token for token in sentence.split() if len(token) >= min_token_length]
        tokenized_sentences.append(tokens)
        all_tokens.extend(tokens)

    # Create the vocabulary with filtered tokens (mapping tokens to indices)
    unique_tokens = set(all_tokens)
    vocab = {token: idx for idx, token in enumerate(unique_tokens)}
    vocab.update({token: len(vocab) + i for i, token in enumerate(['<pad>', '<unk>'])})

    # Create the vocabulary (mapping tokens to indices)
    return vocab

df = pd.read_csv('moviesentiments/data/reviews/cleaned_movie_reviews.csv')
vocab = build_vocab(df['review'])

print(len(vocab))