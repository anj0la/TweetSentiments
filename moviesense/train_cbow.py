"""
File: train_cbow.py

Author: Anjola Aina
Date Modified: October 28th, 2024

This file contains all the necessary functions used to train an RNN-like model.
"""
import torch
import torch.nn as nn
import pandas as pd
from data.datasets import CBOWDataset
from models.cbow import CBOW
from utils.cbow_preprocess import generate_context_target_pairs, tokenize_and_build_vocab
from utils.plot_graphs import plot_loss, plot_pca
from torch import optim
from torch.utils.data import DataLoader

def collate_batch(batch: tuple[list, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts, targets = zip(*batch)
    
    contexts = [torch.tensor(context, dtype=torch.long) for context in contexts]
    targets = [torch.tensor(target, dtype=torch.long) for target in targets]
    
    return contexts, targets

def create_dataloader(file_path: str, batch_size: int) -> tuple[DataLoader, dict]:
    # Preprocess the cleaned dataframe
    df = pd.read_csv(file_path)
    tokenized_data, vocab = tokenize_and_build_vocab(df['review'])
    data = generate_context_target_pairs(tokenized_data, vocab)
    new_df = pd.DataFrame(data, columns=['context', 'target'])
    
    # Create the custom dataset
    dataset = CBOWDataset(new_df)

    # Create the dataloader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    
    return dataloader, vocab

def train_one_epoch(model: CBOW, iterator: DataLoader, optimizer: optim.SGD, device: torch.device, criteron) -> tuple[float, float]:
    # Initialize the epoch loss for every epoch 
    epoch_loss = 0
  
    # Set the model in the training phase
    model.train() 
     
    for batch in iterator:
        # Get the contexts and targets from batch 
        contexts, targets = batch
        
        # Move model input and expected targets to GPU 
        contexts = contexts.to(device)
        targets = targets.to(device)
        
        # Reset the gradients after every batch
        optimizer.zero_grad()   
                
        # Get expected predictions
        outputs = model(contexts).squeeze()
                
        # Compute the loss
        loss = criteron(outputs, targets)
        
        # Backpropagate the loss and compute the gradients
        loss.backward()       
        optimizer.step()    
        
        # Increment the loss
        epoch_loss += loss.item()     
        
    return epoch_loss / len(iterator)
        
def train(file_path: str, model_save_path: str, batch_size: int = 32, n_epochs: int = 10, lr: float = 1e-5) -> None:   
    # Get the dataloader and vocab
    dataloader, vocab = create_dataloader(file_path=file_path, batch_size=batch_size)

    # Get the GPU device (if it exists)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Create the model
    model = CBOW(len(vocab))
        
    # Setup the optimizer and criteron
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteron = nn.CrossEntropyLoss()
    
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Collect all losses
    losses = []
    
    # Train loop
    for epoch in range(n_epochs):
        print(f'Starting epoch {epoch + 1}...')
        loss = train_one_epoch(model, dataloader, optimizer, device, criteron)
        losses.append(round(loss, 2))
        
        # Print loss metrics
        print(f'\t Epoch: {epoch + 1} out of {n_epochs}')
        print(f'\t Train Loss: {loss:.3f}')
        
    # Save the weights
    torch.save(obj=model.embedding.weight, f=model_save_path)
    
    # Visualize and save plots
    plot_loss(x_axis=list(range(1, n_epochs + 1)), train_losses=losses, val_losses=None, figure_path=f'moviesense/figures/cbow/loss_epoch_{n_epochs}_lr_{lr}.png')
    plot_pca(embeddings=model.embedding.weight.detach().cpu().numpy(), vocab=vocab, figure_path='moviesense/figures/cbow/pca_plot.png') 

    

train(file_path='moviesense/data/reviews/cleaned_movie_reviews.csv', model_save_path='moviesense/data/cbow/cbow_model_weights.pth')