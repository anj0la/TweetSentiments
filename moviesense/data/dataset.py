"""
File: dataset.py

Author: Anjola Aina
Date Modified: October 24th, 2024

This file defines a custom Dataset, MovieReviewsDataset, to be used when loading data from the DataLoaders.
"""

import joblib
import numpy as np
import pandas as pd
from utils.preprocess import text_to_sequence
from torch.utils.data import Dataset

class MovieReviewsDataset(Dataset):
    def __init__(self, annotations_file: str, vect_path: str = 'moviesense/data/models/vectorizer.pkl', le_path: str = 'moviesense/data/models/le.pkl', pad_index: int = 0, is_rnn: bool = False) -> None:
        self.reviews = pd.read_csv(annotations_file)
        self.vectorizer = joblib.load(vect_path)
        self.le = joblib.load(le_path)  
        self.is_rnn = is_rnn
      
        # Define vocabulary with padding and unknown indices
        vocab_size = len(self.vectorizer.vocabulary_)
        self.vocabulary = {'<pad>': pad_index, '<unk>': vocab_size + 1, **self.vectorizer.vocabulary_}
        
        # Vectorize text data
        self.vectorized_text = self.vectorizer.transform(self.reviews['review'])
        self.encoded_labels = self.le.transform(self.reviews['sentiment'])
                
    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Gets an item from the reviews.
        
        Note about squeezing the vectorized_text:
        After vectorizing the data, the shape of the text becomes (1, X), where 1 denotes the number of indices to index the vectorized text (initially indexed by single index) and X denotes the number of elements in the vectorized text.
        This means the vectorized text is indexed by two indicies (also known as a two dimensional array).
        
        We squeeze the array to remove the single-dimensional entries from the shape of the vectorized text. This gives us the text with X elements.
     
        Args:
            idx (int): The index of the item to retrieve the vectorized text and corresponding label.

        Returns:
            tuple[np.ndarray, np.ndarray]: The sequence and the corresponding label.
        """
        if not self.is_rnn:
            # MLP case: dense vector representation
            sequence = self.vectorized_text[idx].toarray().squeeze() # Dense matrix
            length = len(sequence) # Not used in MLP but returned for consistency
        else:
            # RNN case: index-based sequence without padding (done in collate_fn function)
            text = self.reviews['review'].iloc[idx]
            sequence = text_to_sequence(text=text, vocab=self.vocabulary, unk_index=self.vocabulary['<unk>'])
            length = len(sequence)
        
        label = self.encoded_labels[idx]
        return sequence, label, length
    
