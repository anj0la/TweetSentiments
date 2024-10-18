"""
File: data_preparation.py
"""
import pandas as pd
import torch

def build_vocab(text: list) -> dict:
    """
    Builds the vocabulary from untokenized text. It is assumed that the text has been preprocessed.

    Args:
        text (pd.Series): The processed text containing all reviews.

    Returns:
        dict: The resulting vocabulary built from the unique tokens in the text.
    """
    tokenized_sentence = []
    for sentence in text: 
        tokenized_sentence.append(sentence.split())
        
    all_tokens = [token for sentence in tokenized_sentence for token in sentence]
    unique_tokens = set(all_tokens)

    # Create the vocabulary (mapping tokens to indices)
    return {token: idx for idx, token in enumerate(unique_tokens)}

def generate_context_target_pairs(text: pd.Series, window_size: int=2) -> tuple[list, dict]:
    """
    Generates the context-target pairs for the CBOW algorithm along with the vocabulary.

    Args:
        text (pd.Series): The processed text containing all reviews.
        window_size (int, optional): The context window size. Defaults to 2.

    Returns:
        tuple: A tuple containing the following elements:
            list: The generated context-target pairs for the model.
            dict: The vocabulary.
    """
    # Build the vocabulary
    vocabulary = build_vocab(text)
    
    # Generate context-target pairs
    context_target_pairs = []
    for sentence in text:
        words = sentence.split()  # Splits words in processed text
        for i in range(len(words)):
            # Getting the context words before the target word at i
            context_words_before = words[max(0, i - window_size): i]  
            # Getting the context words after the target word at i
            context_words_after = words[i + 1: min(len(words),
                                                   i + window_size + 1)]  
            context = context_words_before + context_words_after
            target = words[i]
            # Appending the training sample to the training data
            context_target_pairs.append((context, target)) 
    return context_target_pairs, vocabulary

def word_to_index(word: str, vocab: dict[str, int]) -> int:
    """
    Gets the index of the word in the vocabulary dictionary.

    Args:
        word (str): The word (key) to retrieve its corresponding index in the vocabulary.
        vocab (dict[str, int]): The vocabulary, consisting of all unique words in the document.

    Returns:
        int: The corresponding index (value) of the word (key).
    """
    return vocab.get(word)

def one_hot_encode(word: str, vocab: dict[str, int]) -> torch.Tensor:
    """
    Turns a word into a one hot vector.

    Args:
        word (str): The word to be turned into a one hot vector.
        vocab (dict[str, int]): The vocabulary, consisting of all unique words in the document.

    Returns:
        Tensor: The one hot vector representation of the word.
    """
    index = word_to_index(word, vocab)
    tensor = torch.zeros(1, len(vocab))  # PyTorch assumes everything is in batches, so we set batch size = 1
    tensor[0][index] = 1
    return tensor

def create_one_hot_vectors(input: list[str], vocab: dict[str, int]) -> torch.Tensor:
    """
    Converts a single training example (i.e. a context group) into one hot vectors.

    Args:
        input (list[str]): The training example to be converted into one hot vectors.
        vocab (dict[str, int]): The vocabulary, consisting of all unique words in the document.

    Returns:
        tensor: A tensor containing all one hot vector representations of the input.
    """
    context_vector = []
    for i in range(len(input)):
        one_hot = one_hot_encode(input[i], vocab)
        context_vector.append(one_hot)
    context_tensor = torch.stack(context_vector)
    return context_tensor