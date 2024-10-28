import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import regex as re
import pandas as pd

def tokenize_and_build_vocab(text: pd.Series, min_token_length: int = 2) -> dict:
    """
    Tokenizes text and builds the vocabulary with the specified minimum token length. It is assumed that the text has been preprocessed.

    Args:
        text (list): The processed text containing all reviews.
        min_token_length (int): Minimum length of tokens to keep in the vocabulary. Default is 2.

    Returns:
        dict: The resulting vocabulary built from the unique tokens in the text.
    """
    tokenized_sentence = []
    for sentence in text: 
        # Split the sentence into words and add to tokenized_sentence list
        tokenized_sentence.append(sentence.split())
        
    # Flatten the list of tokenized sentences and filter out single characters if needed
    all_tokens = [token for sentence in tokenized_sentence for token in sentence if len(token) >= min_token_length]
    unique_tokens = set(all_tokens)

    # Create the vocabulary (mapping tokens to indices)
    vocab = {token: idx for idx, token in enumerate(unique_tokens)}
    vocab = {**vocab, '<pad>': len(vocab) + 1, '<unk>': len(vocab) + 2}

    return tokenized_sentence, vocab

def generate_context_target_pairs(text: list, vocab: dict, window_size: int = 2) -> list:
    """
    Generates context-target pairs for the CBOW algorithm, converting each word to its vocabulary index.

    Args:
        text (list): List of tokenized sentences, where each sentence is a list of words.
        vocab (dict): The vocabulary with token-to-index mappings.
        window_size (int, optional): The context window size. Defaults to 2.

    Returns:
        list: Generated context-target pairs with indices for model input.
    """
    # Generate context-target pairs
    context_target_pairs = []
    unk_index = len(vocab)
    
    for sentence in text:
        # Convert each word in the tokenized sentence to a sequence of indices
        sequence = [vocab.get(word, unk_index) for word in sentence]
        
        # Generate context-target pairs from the indexed sequence
        for i, target in enumerate(sequence):
            # Get context indices around target word
            context = sequence[max(0, i - window_size): i] + sequence[i + 1: min(len(sequence), i + window_size + 1)]
            context_target_pairs.append((context, target))
    
    return context_target_pairs

df = pd.read_csv('moviesense/data/reviews/cleaned_movie_reviews.csv')

print(type(df['review']))

tokenized_data, vocab = tokenize_and_build_vocab(df['review'])
X = generate_context_target_pairs(tokenized_data, vocab)

print(X[:1])
