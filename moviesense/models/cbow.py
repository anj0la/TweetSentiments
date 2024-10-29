import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_size: int = 100):
        super(CBOW, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Multi-Layer Perceptron (MLP)
        self.hidden = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, vocab_size) 

    def forward(self, input: torch.Tensor):
        # print('input shape: ', input.shape)
        embeddings = self.embedding(input)
        # print('embedding shape: ', embeddings.shape)
        avg_embeddings = embeddings.mean(dim=1)
        # print('average embedding shape: ', avg_embeddings.shape)
        
        hidden = self.relu(self.hidden(avg_embeddings))
        # print('hidden output: ', hidden.shape)
        
        out = self.fc(hidden)
        
        # print("Output values:", out)
        # print('Classes: ', nn.functional.softmax(out).argmax(dim=1))

        return out