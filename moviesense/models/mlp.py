"""
File: mlp.py

Author: Anjola Aina
Date Modified: October 23rd, 2024

This module defines an MLP-based model for sentiment analysis using PyTorch.

The model has the following structure, given the vocabulary size is 1000 with all default values:
    MLP(
        (embedding): Embedding(1000, 100, padding_idx=0)
        (mlp): ModuleList(
            (0): Linear(in_features=32, out_features=16, bias=True)
            (1): Linear(in_features=16, out_features=1, bias=True)
        )
        (relu): ReLU()
        (dropout): Dropout(p=0.2, inplace=False)
    )
"""
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, vocab_size: int, hidden_dims: list[int] = [128, 64, 32, 16], output_dim: int = 1, dropout: float = 0.2) -> None:
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
            x (torch.Tensor): The input to the MLP.

        Returns:
            torch.Tensor: The output of the model after passing the input through the MLP.
        """
        for i, fc in enumerate(self.mlp):
            if i == 0:
                output = fc(x)
            else:
                output = fc(output)
            if i < len(self.mlp) - 1: # Apply ReLU except on the last layer
                output = self.relu(output)
                # output = self.dropout(output)
                
        # print('output shape: ', output.shape)
                    
        return output