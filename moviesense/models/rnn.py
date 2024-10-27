"""
File: rnn.py

Author: Anjola Aina
Date Modified: October 27th, 2024

This file defines three RNN-based models for sentiment analysis using PyTorch. It utilizes the dynamic multi-layer perceptron (MLP) to increase the complexity of the model.
"""
import torch
import torch.nn as nn
from models.mlp import MLP

class RNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100, pad_index: int = 0, rnn_hidden_dim: int = 256, output_dim: int = 1, n_layers: int = 2, batch_first: bool = True, dropout: float = 0.2, bidirectional: bool = False) -> None:
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
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_index)
                
        # Recurrent Neural Network (RNN) layer
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=rnn_hidden_dim, num_layers=n_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
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
        
    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the RNN.

        Args:
            x (torch.Tensor): The input to the RNN. It is assumed that the input is a numerical input, generated from a bag-of-words model like CountVectorizer, or TfidfVectorizer.
            lengths (torch.Tensor): Lengths of each sequence in the batch.

        Returns:
            torch.Tensor: The raw logits of the model.
        """
        # Embedding layer
        embeddings = self.embedding(input) 
        
        # Print the input shape and embedding output shape
        #print("Input shape (batch size, sequence length):", input.shape)
        #print("Embedding shape:", embeddings.shape)
        
        # Pack the padded embeddings to handle variable lengths
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(input=embeddings, lengths=lengths, batch_first=True, enforce_sorted=False)
        
        # Print the shape of packed embeddings
        #print("Packed embeddings shape:", packed_embeddings.data.shape)
        
        # Initialize the hidden state (number of layers, batch size, hidden size)
        h0 = torch.zeros(self.n_layers * (2 if self.is_bidirectional else 1), input.size(0), self.rnn_hidden_size).to(input.device)
        
        #print('h0 shape: ', h0.shape)
        
        # RNN forward pass
        packed_output, hn = self.rnn(packed_embeddings, h0) # hn has shape (num_layers * 2, batch_size, hidden_size)

        if self.is_bidirectional:
            # Concatenate the last hidden states from both directions ([-2] for forward, [-1] for backward)
            hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # Shape: [batch_size, hidden_size * 2]
        else:
            # For unidirectional, take the last hidden state
            hidden = hn[-1, :, :]  # Shape: [batch_size, hidden_size]

        # MLP forward pass
        output = self.mlp(hidden)
        return output
    
class GRU(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100, pad_index: int = 0, gru_hidden_dim: int = 256, output_dim: int = 1, n_layers: int = 2, batch_first: bool = True, dropout: float = 0.2, bidirectional: bool = False) -> None:
        """
        Builds the multi-layer gated recurrent unit (GRU) RNN neural network used for sentiment analysis.

        Args:
            vocab_size (int): The number of unique words in the vocabulary. It is assumed that a bag-of-words model like CountVectorizer or TfidfVectorizer has been used to generate word embeddings.
            rnn_hidden_dim (int, optionsl): The number of features in the hidden state of the RNN. Defaults to 256.
            output_dim (int, optional): The output layer. Defaults to 1.
            n_layers (int, optional): The number of recurrent layers in the RNN. Default is 2.
            batch_first (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Defaults to True.
            dropout (float, optional): The dropout layer. Defaults to 0.2.
            bidirectional (bool): If True, becomes a bidirectional RNN. Defaults to False.
        """
        super(GRU, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_index)
                
        # Multi-layer Gated recurrent Unit (GRU) RNN layer
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_hidden_dim, num_layers=n_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        
        # MLP layer
        if not bidirectional:
            self.mlp = MLP(vocab_size=gru_hidden_dim, output_dim=output_dim, dropout=dropout)
        else:
            self.mlp = MLP(vocab_size=gru_hidden_dim * 2, output_dim=output_dim, dropout=dropout)
            
        # Number of hidden layers
        self.n_layers = n_layers
        
        # Size of hidden dimension
        self.gru_hidden_size = gru_hidden_dim
        
        # Bidirectional flag
        self.is_bidirectional = bidirectional
        
    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the RNN.

        Args:
            x (torch.Tensor): The input to the RNN. It is assumed that the input is a numerical input, generated from a bag-of-words model like CountVectorizer, or TfidfVectorizer.
            lengths (torch.Tensor): Lengths of each sequence in the batch.

        Returns:
            torch.Tensor: The raw logits of the model.
        """
        # Embedding layer
        embeddings = self.embedding(input) 
        
        # Pack the padded embeddings to handle variable lengths
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(input=embeddings, lengths=lengths, batch_first=True, enforce_sorted=False)
        
        # Initialize the hidden state (number of layers, batch size, hidden size)
        h0 = torch.zeros(self.n_layers * (2 if self.is_bidirectional else 1), input.size(0), self.gru_hidden_size).to(input.device)
                
        # GRU forward pass
        packed_output, hn = self.gru(packed_embeddings, h0)

        if self.is_bidirectional:
            # Concatenate the last hidden states from both directions ([-2] for forward, [-1] for backward)
            hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # Shape: [batch_size, hidden_size * 2]
        else:
            # For unidirectional, take the last hidden state
            hidden = hn[-1, :, :]  # Shape: [batch_size, hidden_size]

        # MLP pass
        output = self.mlp(hidden)
        return output
    
class LSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100, pad_index: int = 0, lstm_hidden_dim: int = 256, output_dim: int = 1, n_layers: int = 2, batch_first: bool = True, dropout: float = 0.2, bidirectional: bool = False) -> None:
        """
        Builds the multi-layer gated recurrent unit (GRU) RNN neural network used for sentiment analysis.

        Args:
            vocab_size (int): The number of unique words in the vocabulary. It is assumed that a bag-of-words model like CountVectorizer or TfidfVectorizer has been used to generate word embeddings.
            rnn_hidden_dim (int, optionsl): The number of features in the hidden state of the RNN. Defaults to 256.
            output_dim (int, optional): The output layer. Defaults to 1.
            n_layers (int, optional): The number of recurrent layers in the RNN. Default is 2.
            batch_first (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Defaults to True.
            dropout (float, optional): The dropout layer. Defaults to 0.2.
            bidirectional (bool): If True, becomes a bidirectional RNN. Defaults to False.
        """
        super(LSTM, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_index)
                
        # Long-term short memory (LSTM) layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=n_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        
        # MLP layer
        if not bidirectional:
            self.mlp = MLP(vocab_size=lstm_hidden_dim, output_dim=output_dim, dropout=dropout)
        else:
            self.mlp = MLP(vocab_size=lstm_hidden_dim * 2, output_dim=output_dim, dropout=dropout)
            
        # Number of hidden layers
        self.n_layers = n_layers
        
        # Size of hidden dimension
        self.lstm_hidden_size = lstm_hidden_dim
        
        # Bidirectional flag
        self.is_bidirectional = bidirectional
        
    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the RNN.

        Args:
            x (torch.Tensor): The input to the RNN. It is assumed that the input is a numerical input, generated from a bag-of-words model like CountVectorizer, or TfidfVectorizer.
            lengths (torch.Tensor): Lengths of each sequence in the batch.

        Returns:
            torch.Tensor: The raw logits of the model.
        """
        # Embedding layer
        embeddings = self.embedding(input) 
        
        # Pack the padded embeddings to handle variable lengths
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(input=embeddings, lengths=lengths, batch_first=True, enforce_sorted=False)
        
        # Initialize the hidden state and cell state (number of layers, batch size, hidden size)
        h0 = torch.zeros(self.n_layers * (2 if self.is_bidirectional else 1), input.size(0), self.lstm_hidden_size).to(input.device)
        c0 = h0
    
        # LSTM forward pass
        packed_output, (hn, cn) = self.lstm(packed_embeddings, (h0, c0))

        if self.is_bidirectional:
            # Concatenate the last hidden states from both directions ([-2] for forward, [-1] for backward)
            hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # Shape: [batch_size, hidden_size * 2]
        else:
            # For unidirectional, take the last hidden state
            hidden = hn[-1, :, :]  # Shape: [batch_size, hidden_size]
            
        # MLP forward pass
        output = self.mlp(hidden)
        return output