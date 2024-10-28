import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128, embedding_dim: int = 100):
        super(CBOW, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Multi-Layer Perceptron (MLP)
        self.hidden = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, vocab_size) 

    def forward(self, x: torch.Tensor):
        out = self.embedding(x)  
        out = out.mean(dim=1)  
        out = self.relu(self.hidden(out))
        out = self.fc(out)

        return out