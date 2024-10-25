"""
File: mlp.py

Author: Anjola Aina
Date Modified: October 24th, 2024

This module defines an MLP-based model for sentiment analysis using PyTorch.

The model has the following structure, given the vocabulary size is 169548 with all default values:
    MLP(
        (mlp): ModuleList(
            (0): Linear(in_features=169548, out_features=128, bias=True)
            (1): Linear(in_features=128, out_features=64, bias=True)
            (2): Linear(in_features=64, out_features=32, bias=True)
            (3): Linear(in_features=32, out_features=16, bias=True)
            (4): Linear(in_features=16, out_features=1, bias=True)
        )
        (relu): ReLU()
        (dropout): Dropout(p=0.2, inplace=False)
    )
"""
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, vocab_size: int, hidden_dims: list[int] = [128, 64, 32, 16], output_dim: int = 1, dropout: float = 0.2) -> None:
        """
        Represents a multilayer perceptron (MLP) consisting of a dynamic number of hidden layers to increase the complexity and depth of the MLP.

        Args:
            vocab_size (int): The number of unique words in the vocabulary. It is assumed that a bag-of-words model like CountVectorizer or TfidfVectorizer has been used to generate word embeddings.
            hidden_dims (list[int], optional): The number of hidden layers. Defaults to [128, 64, 32, 16].
            output_dim (int, optional): The output layer. Defaults to 1.
            dropout (float, optional): The dropout layer. Defaults to 0.2.
        """
        super(MLP, self).__init__()
                
        # MLP layer
        self.mlp = nn.ModuleList(
            [nn.Linear(vocab_size, hidden_dims[0])]            
        ).extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]) if i < len(hidden_dims) - 1 else nn.Linear(hidden_dims[i], output_dim) for i in range(len(hidden_dims))])
        
        # Activation function (for hidden layers)
        self.relu = nn.ReLU()
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the MLP.
        
        Args:
            x (torch.Tensor): The input to the MLP. It is assumed that the input is a numerical input, generated from a bag-of-words model like CountVectorizer, or TfidfVectorizer.

        Returns:
            torch.Tensor: The raw logits of the model.
        """
        for i, fc in enumerate(self.mlp):
            if i == 0:
                output = fc(x)
            else:
                output = fc(output)
            if i < len(self.mlp) - 1: # Apply ReLU except on the last layer
                output = self.relu(output)
                output = self.dropout(output)
                                    
        return output