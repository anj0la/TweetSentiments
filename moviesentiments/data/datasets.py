"""
File: dataset.py

Author: Anjola Aina
Date Modified: October 28th, 2024

This file defines a custom Dataset, MovieReviewsDataset, to be used when loading data from the DataLoaders.
"""

import joblib
import numpy as np
import pandas as pd
import torch
from utils.preprocess import word_to_idx, build_vocab
from torch.utils.data import Dataset

class MovieReviewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, le_path: str = 'moviesense/data/models/le.pkl') -> None:
        self.df = df
        self.vocab = build_vocab(self.df['review'])
        self.le = joblib.load(le_path)  
        self.encoded_labels = self.le.transform(self.df['sentiment'])
                
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        text = self.df['review'].iloc[idx]
        sequence = word_to_idx(text=text, vocab=self.vocab, unk_index=self.vocab['<unk>'])
        length = len(sequence)
        
        label = self.encoded_labels[idx]
        return sequence, label, length
    
class CBOWDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        # Assuming 'context' contains lists of context word indices
        self.contexts = [torch.tensor(context, dtype=torch.long) for context in df['context'].tolist()]
        self.targets = torch.tensor(df['target'].values, dtype=torch.long)
                
    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:        
        return self.contexts[idx], self.targets[idx]