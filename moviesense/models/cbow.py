"""
File: cbow.py

Author: Anjola Aina
Date Modified: October 18th, 2024

Description:

This file contains the CBOW class which is used to implement the continous bag of words (CBOW) algorithm.
"""
import torch
import torch.nn as nn

class CBOW(nn.Module):
    """
    Implements the CBOW model. It inherits all attributes from its base class, the Module class.
    It creates the embedding and MLP layers, along with the ReLU activiation function.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_size: int = 200):
        super(CBOW, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Multi-Layer Perceptron (MLP)
        self.hidden = nn.Linear(embedding_dim, hidden_size) # Linear = fully connected layer
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, vocab_size) # Linear = fully connected layer

    def forward(self, x):
        """
        Implements the forward pass for the CBOW architecture.

        Args:
            x (Tensor): the input to the CBOW model.

        Returns:
            Any: the log probability of the model (i.e., the prediction).
        """
        # 4D result vector (batch_size, seq_len, vocab_size, dim_size)
        embeddings = self.embedding(x)  
        # Get average embeddings across vocabulary
        average_embeddings = torch.mean(embeddings, dim=2)  
        hidden_output = self.relu(self.hidden(average_embeddings))
        output = self.output(hidden_output)

        return output[0]