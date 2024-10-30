"""
File: logistic_regression.py

Author: Anjola Aina
Date Modified: October 29th, 2024

This file defines an Logisitic Regression model for sentiment analysis using PyTorch.

The model has the following structure, given the vocabulary size is 169548 with all default values:
"""
import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100, output_dim: int = 1, dropout: float = 0.2, pad_index: int = 0) -> None:
        """
        Represents a multilayer perceptron (MLP) consisting of a dynamic number of hidden layers to increase the complexity and depth of the MLP.

        Args:
            vocab_size (int): The number of unique words in the vocabulary. It is assumed that a bag-of-words model like CountVectorizer or TfidfVectorizer has been used to generate word embeddings.
            hidden_dims (list[int], optional): The number of hidden layers. Defaults to [128, 64, 32, 16].
            output_dim (int, optional): The output layer. Defaults to 1.
            dropout (float, optional): The dropout layer. Defaults to 0.2.
        """
        super(LogisticRegression, self).__init__()
                
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_index)
        
        # Linear layer
        self.fc = nn.Linear(embedding_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input: torch.Tensor, lengths: int = None) -> torch.Tensor:
        """
        Implements the forward pass for the MLP.
        
        Args:
            input (torch.Tensor): The input to the MLP. It is assumed that the input is a numerical input, generated from a bag-of-words model like CountVectorizer, or TfidfVectorizer.

        Returns:
            torch.Tensor: The raw logits of the model.
        """
        output = self.fc(input)
        output = self.dropout(output)
        return output