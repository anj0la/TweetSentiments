"""
File: rnn.py

Author: Anjola Aina
Date Modified: October 25th, 2024

This file defines an RNN-based model for sentiment analysis using PyTorch.

The model has the following structure, given the vocabulary size is 169548 with all default values:
    RNN(
        (rnn): RNN(169548, 256, num_layers=2, batch_first=True, dropout=0.2)
        (mlp): MLP(
            (mlp): ModuleList(
            (0): Linear(in_features=256, out_features=128, bias=True)
            (1): Linear(in_features=128, out_features=64, bias=True)
            (2): Linear(in_features=64, out_features=32, bias=True)
            (3): Linear(in_features=32, out_features=16, bias=True)
            (4): Linear(in_features=16, out_features=1, bias=True)
            )
            (relu): ReLU()
            (dropout): Dropout(p=0.2, inplace=False)
        )
    )
"""
import torch
import torch.nn as nn
from mlp import MLP

class RNN(nn.Module):
    def __init__(self, vocab_size: int, rnn_hidden_dim: int = 256, output_dim: int = 1, n_layers: int = 2, batch_first: bool = True, dropout: float = 0.2, bidirectional: bool = False) -> None:
        """
        Represents a recurrent neural network (RNN) used for sentiment analysis.

        Args:
            vocab_size (int): The number of unique words in the vocabulary. It is assumed that a bag-of-words model like CountVectorizer or TfidfVectorizer has been used to generate word embeddings.
            rnn_hidden_dim (int, optionsl): The number of features in the hidden state of the RNN. Defaults to 256.
            output_dim (int, optional): The output layer. Defaults to 1.
            n_layers (int, optional): The number of recurrent layers in the RNN. Default is 2.
            batch_first (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Defaults to True.
            dropout (float, optional): The dropout layer. Defaults to 0.2.
            bidirectional (bool): If True, becomes a bidirectional RNN. Defaults to False.
        """
        super(RNN, self).__init__()
                
        # Recurrent Neural Network (RNN) layer
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=rnn_hidden_dim, num_layers=n_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        
        # MLP layer
        if not bidirectional:
            self.mlp = MLP(vocab_size=rnn_hidden_dim, output_dim=output_dim, dropout=dropout)
        else:
            self.mlp = MLP(vocab_size=rnn_hidden_dim * 2, output_dim=output_dim, dropout=dropout)
            
        # Number of hidden layers
        self.n_layers = n_layers
        
        # Size of hidden dimension
        self.rnn_hidden_size = rnn_hidden_dim
        
        # Bidirectional flag
        self.is_bidirectional = bidirectional
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the RNN.

        Args:
            x (torch.Tensor): The input to the RNN. It is assumed that the input is a numerical input, generated from a bag-of-words model like CountVectorizer, or TfidfVectorizer.
            lengths (torch.Tensor): Lengths of each sequence in the batch.

        Returns:
            torch.Tensor: The raw logits of the model.
        """
        # Pack the padded input sequence to handle variable lengths
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Initialize the hidden state (number of layers, batch size, hidden size)
        h0 = torch.zeros(self.n_layers * (2 if self.is_bidirectional else 1), x.size(0), self.rnn_hidden_size).to(x.device)
        
        packed_output, hn = self.rnn(packed_input, h0) # hn has shape (num_layers * 2, batch_size, hidden_size)

        if self.is_bidirectional:
            # Concatenate the last hidden states from both directions ([-2] for forward, [-1] for backward)
            hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # Shape: [batch_size, hidden_size * 2]
        else:
            # For unidirectional, take the last hidden state
            hidden = hn[-1, :, :]  # Shape: [batch_size, hidden_size]

        output = self.mlp(hidden)
        return output

rnn = RNN(169548)
print(rnn)