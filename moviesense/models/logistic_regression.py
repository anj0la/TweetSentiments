"""
File: logistic_regression.py

Author: Anjola Aina
Date Modified: October 18th, 2024

Description:

This file contains the LogisticRegression class which is used to implement a binary classifier with a sigmoid activation function.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisiticRegression:
    """
    A class to represent the implementation of a logistic regression with a sigmoid activation function.
    """
    def __init__(self, lr: float = 0.1, epochs: int = 10, batch_size: int = 64, decay_factor: float = 0.1, lr_step: int = 10, reg_lambda: float = 0.0) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None # Bias
        self.batch_size = batch_size
        self.decay_factor = decay_factor
        self.lr_step = lr_step
        self.reg_lambda = reg_lambda
    
    def _initialize_weights(self, n_features) -> None:
        """ 
        Initializes the weights in the binary classifer by assigning fixed values to the weights.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0.5
        
    def _forward_pass(self, x):
        z = self.bias + np.dot(x, self.weights)
        return self._sigmoid(z)
        
    def _update_weights(self, X, y, y_hat):
        m = X.shape[0]
        d_weight = (1 / m) * np.dot(X.T, (y_hat - y))
        d_bias = (1 / m) * np.sum(y_hat - y)
        self.weights -= self.lr * d_weight
        self.bias -= self.lr * d_bias
            
    def _loss_function(self, y, y_hat):
        # Clip y_hat to avoid log(0)
        y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
        m = y.shape[0]
        
        cross_entropy_loss = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        l2_reg = (self.reg_lambda / (2 * m)) * np.sum(np.square(self.weights))
        return cross_entropy_loss + l2_reg
       
    
    def _evaluate(self, X_val, y_val):
        # Forward pass on entire validation set
        y_hat = self._forward_pass(X_val)
        
        # Compute the loss
        loss = self._loss_function(y_val, y_hat)
        
        # Predicted labels
        predicted_labels = [1 if pred >= 0.5 else 0 for pred in y_hat]
        
        # Compute accuracy
        accuracy = accuracy_score(y_true=y_val, y_pred=predicted_labels)
        
        return loss, accuracy
    
    def _sigmoid(self, z):
        """
        Implements the sigmoid function, an activiation function that changes the weighted sum z to a probability.

        Args:
            z (Any): The weighted sum.

        Returns:
            Any: the predicted probability of z (either 0 or 1)
        """
        return 1 / (1 + np.exp(-z))
    
    def _plot_accuracy(self, x_axis, val_accuracy):
        """
        This function plots a graph that visualizes how the loss decreases over the epochs. That is, as the epochs increase, the loss decreases.

        Args:
            list_epochs (list): all the epochs (iterations)
            list_total_loss (list): all the total losses per epoch
        """
        fig, ax = plt.subplots()
        
        # Plot validation accuracy
        ax.plot(x_axis, val_accuracy) 
        
        # Set labels and title
        ax.set_xlabel('Number of Epochs')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy as a Function of Epochs')

        # Save the plot
        plt.savefig(f'moviesense/figures/logistic_regression/accuracy_epoch_{len(x_axis)}_lr_0.01.png')

        # plt.show()
        
    def _plot_loss(self, x_axis, train_losses, val_losses):
        """
        This function plots a graph that visualizes how the loss decreases over the epochs
        for both the training and validation sets.

        Args:
            x_axis (list): All the epochs (iterations)
            train_losses (list): All the total training losses per epoch
            val_losses (list): All the total validation losses per epoch
            y_label (str): Label for the y-axis (e.g., 'Loss')
            lr (float): Learning rate used in training, included in the title and filename
        """
        fig, ax = plt.subplots()
        
        # Plot training losses
        ax.plot(x_axis, train_losses, label='Training Loss', color='blue')
        
        # Plot validation losses
        ax.plot(x_axis, val_losses, label='Validation Loss', color='orange', linestyle='--')

        # Set labels and title
        ax.set_xlabel('Number of Epochs')
        ax.set_ylabel('Total Loss')
        ax.set_title(f'Loss as a Function of Epochs')

        # Add legend to distinguish between train/validation
        ax.legend()

        # Save the plot
        plt.savefig(f'moviesense/figures/logistic_regression/loss_epoch_{len(x_axis)}_lr_0.01.png')
        
        # plt.show()
        
    def _save_model(self):
        np.save('movie_sense/data/models/logistic_regression/weights.npy', self.weights)
        np.save('movie_sense/data/models/logistic_regression/bias.npy', np.array(self.bias))

    def fit(self, X_train, y_train):
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Convert to dense arrays to work with model
        X_train, X_val = X_train.toarray(), X_val.toarray()
        
        self._initialize_weights(X_train.shape[1])
        all_train_losses = []
        all_val_losses = []
        all_val_accuracy = []
        num_batches = X_train.shape[0] // self.batch_size
        
        for epoch in range(self.epochs):
            
            print(f'Starting epoch {epoch + 1}...')
            
            # Manual learning rate scheduler
            if epoch % self.lr_step == 0 and epoch != 0:
                self.lr *= self.decay_factor
                print(f'Reduced learning rate to {self.lr}')
            
            # Generate batch indices and shuffle them
            batch_indices = np.arange(num_batches)
            np.random.shuffle(batch_indices)
            
            # Reset loss and labels after every epoch
            epoch_loss = 0
            total_correct = 0
            total_samples = 0
            
            for i in batch_indices:
                # Get the mini-batch
                start_i = i * self.batch_size
                end_i = start_i + self.batch_size
                X_batch = X_train[start_i:end_i]
                y_batch = y_train[start_i:end_i]
                
                # Forward pass
                y_hat = self._forward_pass(X_batch)
                
                # Compute predictions and accuracy for the current batch
                batch_predicted_labels = [1 if pred >= 0.5 else 0 for pred in y_hat]
                total_correct += sum(1 for true, pred in zip(y_batch, batch_predicted_labels) if true == pred)
                total_samples += len(y_batch)
                
                # Compute the loss for current batch
                batch_loss = self._loss_function(y_batch, y_hat)
                epoch_loss += batch_loss
                
                # Update the weights for current batch
                self._update_weights(X_batch, y_batch, y_hat)
                
            # Compute average loss and accuracy        
            avg_loss = epoch_loss / num_batches
            train_accuracy = total_correct / total_samples

            # Get validaation loss and accuracy
            val_loss, val_accuracy = self._evaluate(X_val, y_val)

            # Append train, val losses and accurary
            all_train_losses.append(avg_loss)
            all_val_losses.append(val_loss)
            all_val_accuracy.append(round(val_accuracy * 100, 2))
            
            # Print train and val metrics
            print(f'\t Epoch: {epoch + 1} out of {self.epochs}')
            print(f'\t Train Loss: {avg_loss:.3f} | Train Acc: {train_accuracy * 100:.2f}%')
            print(f'\t Valid Loss: {val_loss:.3f} | Valid Acc: {val_accuracy * 100:.2f}%')
            
        # Visualize (and save) plots
        x_axis = list(range(1, self.epochs + 1))
        self._plot_loss(x_axis, all_train_losses, all_val_losses)  
        self._plot_accuracy(x_axis, all_val_accuracy)  
        
        # Save the weights and bias to be used for predictions
        # self._save_model()

    def predict(self, X):
        """
        Predicts the probability (output either 0 or 1) for a given input X, by using the sigmoid function.
        As the sigmoid function may give a decimal value, we use np.round so that values over 0.5 (inclusive) are rounded up to 1,
        and values less than 0.5 (exclusive) are rounded down to 0.

        Args:
            X (ndarray): The input to make a prediction on.

        Returns:
            int: The predicted probability of the input (either 0 or 1).
        """
        z = np.dot(X, self.weights) + self.bias
        return np.round(self._sigmoid(z))
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
