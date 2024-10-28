"""
File: dataset.py

Author: Anjola Aina
Date Modified: October 24th, 2024

This file defines a custom Dataset, MovieReviewsDataset, to be used when loading data from the DataLoaders.
"""

import joblib
import numpy as np
import pandas as pd
import torch
from utils.preprocess import text_to_sequence
from torch.utils.data import Dataset

class CBOWDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df
                
    def __len__(self) -> int:
        return len(self.data)

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
        context = self.data['context'].iloc(idx)
        target = self.data['target'].iloc(idx)
        
        return context, target
    
