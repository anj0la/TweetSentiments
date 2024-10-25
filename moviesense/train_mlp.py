"""
File: train.py

Author: Anjola Aina
Date Modified: October 23rd, 2024

This file contains all the necessary functions used to train the MLP model.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataset import MovieReviewsDataset
from models.mlp import MLP
from utils.plot_graphs import plot_accuracy, plot_loss
from utils.preprocess import preprocess
from torch import optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score

def collate_batch(batch: tuple[list[int], int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of data for the DataLoader.

    This function takes a batch of sequences, labels, and lengths, converts them to tensors, 
    and pads the sequences to ensure they are of equal length. This is useful for feeding data 
    into models that require fixed-length inputs, such as LSTM models.

    Args:
        batch (list of tuples): A list where each element is a tuple containing two elements:
            sequences (list of int): The sequence of token ids representing a piece of text.
            labels (int): The label corresponding to the sequence.

    Returns:
        tuple: A tuple containing two elements:
            padded_sequences (torch.Tensor): A tensor of shape (batch_size, max_sequence_length) containing the padded sequences.
            labels (torch.Tensor): A tensor of shape (batch_size,) containing the labels.
    """
    encoded_sequences, encoded_labels = zip(*batch)
        
    # Converting the sequences, labels and sequence length to Tensors
    encoded_sequences = [torch.tensor(seq, dtype=torch.float) for seq in encoded_sequences]
    encoded_labels = torch.tensor(encoded_labels, dtype=torch.float)
        
    # Padding sequences
    padded_encoded_sequences = nn.utils.rnn.pad_sequence(encoded_sequences, batch_first=True, padding_value=0)    
    
    return padded_encoded_sequences, encoded_labels

def create_dataloaders(file_path: str, batch_size: int, train_split: float, val_split: float) -> tuple[DataLoader, DataLoader]:
    """
    Creates custom datasets and dataloaders for training and testing.

    Args:
        file_path (str): The path to the processed CSV file containing the data.
        batch_size (int): The size of the batches for the dataloaders. Default is 64.
        train_split (float): The proportion of the data to use for training. Default is 0.8.

    Returns:
        tuple: A tuple containing:
            - DataLoader: The dataloader for the training dataset.
            - DataLoader: The dataloader for the testing dataset.
    """
    # Create the custom dataset
    dataset = MovieReviewsDataset(file_path)
    
    # Calculate sizes for training, validation, and test sets
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size  # Ensure all data is accounted for

    # Split the dataset into training, validation, and testing sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create dataloaders for the training, validation, and testing sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    return train_dataloader, val_dataloader, test_dataloader, dataset

def train_one_epoch(model: MLP, iterator: DataLoader, optimizer: optim.SGD, device: torch.device) -> tuple[float, float]:
    """
    Trains the model for one epoch.

    Args:
        model (MLP): The model to be trained.
        iterator (DataLoader): The DataLoader containing the training data.
        optimizer (optim.SGD): The optimizer used for updating model parameters.

    Returns:
        tuple: A tuple containing:
            - float: The average loss over the epoch.
            - float: The average accuracy over the epoch.
    """
    # Initialize the epoch loss for every epoch 
    epoch_loss = 0
    count = 0
    all_predictions = []
    all_labels = []
    
    # Set the model in the training phase
    model.train() 
     
    for batch in iterator:
        # Get the padded sequences and labels from batch 
        padded_sequences, labels = batch
        # labels = labels.type(torch.LongTensor) # Casting to long
        
        # Move input and expected label to GPU
        padded_sequences = padded_sequences.to(device)
        labels = labels.to(device)
        
        # Reset the gradients after every batch
        optimizer.zero_grad()   
                
        # Get expected predictions
        predictions = model(padded_sequences).squeeze()
        
        # if count == 0:
        #     print('predictions shape: ', predictions.shape)
        #     print('labels shape: ', labels.shape)
        #     print('predictions: ', predictions)
        #     print('labels: ', labels)

        #print('predictions shape: ', predictions.shape)
        #print('predictions: ', predictions)

        #print('labels shape: ', labels.shape)
        #print('labels: ', labels)
                
        # Compute the loss
        loss = F.binary_cross_entropy_with_logits(predictions, labels)   
        
        # Backpropagate the loss and compute the gradients
        loss.backward()       
        optimizer.step()    
        
        # Increment the loss
        epoch_loss += loss.item()       
        
        # Store predictions and labels for accuracy calculation
        predicted_labels = torch.round(F.sigmoid(predictions))  # Get binary predictions
        all_predictions.append(predicted_labels.detach().cpu())
        all_labels.append(labels.detach().cpu())
        
        count += 1
        
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # Compute accuracy over the entire epoch
    correct = (all_predictions == all_labels).float().sum()
    accuracy = correct / len(all_labels)
        
    return epoch_loss / len(iterator), accuracy.item()

def evaluate_one_epoch(model: MLP, iterator: DataLoader, device: torch.device) -> tuple[float, float]:
    """
    Evaluates the model on the validation/test set.

    Args:
        model (LSTM): The model to be evaluated.
        iterator (DataLoader): The DataLoader containing the validation/test data.
        device (torch.device): The device to train the model on.

    Returns:
        tuple: A tuple containing:
            - float: The average loss over the validation/test set.
            - float: The average accuracy over the validation/test set.
    """
    # Initialize the epoch loss for every epoch 
    epoch_loss = 0
    all_predictions = []
    all_labels = []
    
    # Set the model in evaluation mode (disables dropout, etc.)
    model.eval()
    with torch.no_grad(): # Deactivates autograd (no gradients needed)
        
        for batch in iterator:
            
            # Get the padded sequences and labels from batch 
            padded_sequences, labels = batch
            # labels = labels.type(torch.LongTensor) # Casting to long
                        
            # Move sequences and expected labels to GPU
            padded_sequences = padded_sequences.to(device)
            labels = labels.to(device)
            
            # Get expected predictions
            predictions = model(padded_sequences).squeeze()
            
            # Compute the loss
            loss = F.binary_cross_entropy_with_logits(predictions, labels)     
            
            # Increment the loss
            epoch_loss += loss.item()       
            
            # Store predictions and labels for accuracy calculation
            predicted_labels = torch.round(F.sigmoid(predictions)) # Get binary predictions
            all_predictions.append(predicted_labels.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # Compute accuracy over the entire epoch
    correct = (all_predictions == all_labels).float().sum()
    accuracy = correct / len(all_labels)
    
    return epoch_loss / len(iterator), accuracy.item()
        
def train(input_file_path: str, cleaned_file_path: str, model_save_path: str, train_ratio: int = 0.6, val_ratio: int = 0.2, batch_size: int = 32, n_epochs: int = 10, 
               lr: float = 1e-5, weight_decay: float = 1e-5) -> None:
    """
    Trains an MLP model used for sentiment analysis.
    """
    # Preprocess the file (if not already preprocessed)
    if not os.path.exists(cleaned_file_path):
        preprocess(file_path=input_file_path, output_file_path=cleaned_file_path)
        
    # Get the training, validation and testing dataloaders
    train_dataloader, val_dataloader, test_dataloader, dataset = create_dataloaders(
        file_path=cleaned_file_path, batch_size=batch_size, train_split=train_ratio, val_split=val_ratio
    )
    
    # Get the GPU device (if it exists)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Create the model
    model = MLP(vocab_size=len(dataset.vocabulary)).to(device)
    print(model)
    
    # Setup the optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Collecting train and val losses and val accuracy
    train_losses = []
    val_losses = []
    val_accuracy_list = []
    
    # Initalizing best loss and clearing GPU cache
    best_val_loss = float('inf')
    torch.cuda.empty_cache()

    # Training / testing model
    for epoch in range(n_epochs):
        print(f'Starting epoch {epoch + 1}...')
        
        # Train the model
        train_loss, train_accurary = train_one_epoch(model, train_dataloader, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate the model
        val_loss, val_accuracy = evaluate_one_epoch(model, val_dataloader, device)
        val_losses.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(obj=model.state_dict(), f=model_save_path)
        
        # Print train / valid metrics
        print(f'\t Epoch: {epoch + 1} out of {n_epochs}')
        print(f'\t Train Loss: {train_loss:.3f} | Train Acc: {train_accurary * 100:.2f}%')
        print(f'\t Valid Loss: {val_loss:.3f} | Valid Acc: {val_accuracy * 100:.2f}%')
        
    # Visualize and save plots
    plot_loss(x_axis=list(range(1, n_epochs + 1)), train_losses=train_losses, val_losses=val_losses, figure_path=f'moviesense/figures/mlp/loss_epoch_{n_epochs}_lr_{lr}.png')
    plot_accuracy(x_axis=list(range(1, n_epochs + 1)), val_accuracy=val_accuracy_list, figure_path=f'moviesense/figures/mlp/accuracy_epoch_{n_epochs}_lr_{lr}.png')

    # Evaluate model on testing set
    test_accuracy, precision, recall, f1 = evaluate(model, test_dataloader, device)
        
    # Print test metrics
    print(f'Test Acc: {test_accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')    
    print(f'Recall: {recall * 100:.2f}%')    
    print(f'F1 Score: {f1 * 100:.2f}%')   
    
def evaluate(model: MLP, iterator: DataLoader, device: torch.device) -> tuple[float, float, float, float]:
    all_predictions = []
    all_labels = []
    
    # Set the model in evaluation mode (disables dropout, etc.)
    model.eval()
    with torch.no_grad(): # Deactivates autograd (no gradients needed)
        
        for batch in iterator:
            # Get the padded sequences and labels from batch 
            padded_sequences, labels = batch
            labels = labels.type(torch.LongTensor) # Casting to long
                        
            # Move sequences and expected labels to GPU
            padded_sequences = padded_sequences.to(device)
            labels = labels.to(device)
            
            # Get expected predictions
            predictions = model(padded_sequences).squeeze(1)
            
            # Apply the sigmoid function and round to get binary predictions
            predicted_labels = torch.round(F.sigmoid(predictions))
            
            # Store predictions and labels for metric calculations
            all_predictions.append(predicted_labels.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # Compute accuracy
    correct = (all_predictions == all_labels).float().sum()
    accuracy = correct / len(all_labels)
    
     # Convert tensors to numpy arrays for sklearn metric calculations
    all_predictions_np = all_predictions.numpy()
    all_labels_np = all_labels.numpy()
    
    # Compute precision, recall, and F1-score using sklearn
    precision = precision_score(all_labels_np, all_predictions_np)
    recall = recall_score(all_labels_np, all_predictions_np)
    f1 = f1_score(all_labels_np, all_predictions_np)
    
    return accuracy, precision, recall, f1
    
##### Running the code #####
train(input_file_path='moviesense/data/reviews/IMDB Dataset.csv', cleaned_file_path='moviesense/data/reviews/cleaned_movie_reviews.csv', model_save_path='moviesense/data/models/mlp/mlp_saved_state.pt')