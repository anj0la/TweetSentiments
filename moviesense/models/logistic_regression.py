"""
File: logistic_regression.py

Author: Anjola Aina
Date Modified: October 18th, 2024

Description:

This file contains the LogisticRegression class which is used to implement a binary classifier with a sigmoid activation function.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisiticRegression:
    """
    A class to represent the implementation of a logistic regression with a sigmoid activation function.
    
    Attributes:
        - learning_rate - the learning rate of the classifier
        - n_iterations - the number of iterations to go through all of the training examples ( known as epochs) 
        - weights - the parameters of the model, adjusted to get the desired probability
        - w0 - an extra parameter, adjusted to get the desired probability
    """
    def __init__(self, lr: float = 0.1, epochs: int = 200, batch_size: int = 64) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None # Bias
        self.batch_size = batch_size
    
    def _initialize_weights(self, n_features) -> None:
        """ 
        Initializes the weights in the binary classifer by assigning fixed values to the weights.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0.5

    def fit(self, X_train, y_train, X_val, y_val):
        self._initialize_weights(X_train.shape[1])
        all_train_losses = []
        all_val_losses = []
        all_val_accurary = []
        num_batches = X_train.shape[0] // self.batch_size
        
        for epoch in range(self.epochs):
            
            print(f'Starting epoch {epoch + 1}...')
            
            # Generate batch indices and shuffle them
            batch_indices = np.arange(num_batches)
            np.random.shuffle(batch_indices)
            
            # Reset loss after every epoch
            epoch_loss = 0
            
            for i in batch_indices:
                # Get the mini-batch
                start_i = i * self.batch_size
                end_i = start_i + self.batch_size
                X_batch = X_train[start_i:end_i]
                y_batch = y_train[start_i:end_i]
                
                # Forward pass
                y_hat = self._forward_pass(X_batch)
                
                # Compute the loss for current batch
                batch_loss = self._loss_function(y_batch, y_hat)
                epoch_loss += batch_loss
                
                # Update the weights for current batch
                self._update_weights(X_batch, y_batch, y_hat)
                
            val_loss, val_accurary = self._evaluate(X_val, y_val)
            
            # Get predicted labels
            predicted_labels = [1 if pred >= 0.5 else 0 for pred in y_hat]
        
            # Compute average loss and accuracy        
            avg_loss = epoch_loss / num_batches
            train_accuracy = accuracy_score(y_true=y_train, y_pred=predicted_labels)

            # Append train, val losses and accurary
            all_train_losses.append(avg_loss)
            all_val_losses.append(val_loss)
            all_val_accurary.append(val_accurary)
            
            print(f'Epoch {epoch + 1} finished with average loss: {avg_loss}')

            
            # # Forward pass (make a prediction)
            # y_hat = self._forward_pass(X_train)
            
            # print('WE MADE A FORWARD PASS!')
            
            # #print(f'type of y_train: {type(y_train)}, type of y_hat: {type(y_hat)}')
            # #print(f'y_train[:5]: {y_train[:5]}, y_hat[:5]: {y_hat[:5]}')
            # #print(f'y_train type: {y_train.dtype}, y_hat type: {y_hat.dtype}')

            # # Calculate the loss
            # loss = self._loss_function(y_train, y_hat)
            
            # print('WE CALCLUATED THE LOSS!')
            
            # # Backward pass and update weights
            # self._update_weights(X_train, y_train, y_hat)
            
            # print('WE UPDATED THE WEIGHTS!')
                
            # Print loss metrics
            # print(f'Epoch {epoch + 1} / {self.epochs}\nCurrent loss: {loss}')
            # # print(f'Weights : {self.weights}')
            
            # all_train_losses.append(loss)
            
        # Visualize (and save) plots
        x_axis = list(range(1, self.epochs + 1))
        self._plot_loss(x_axis, all_train_losses, all_val_losses, self.lr)  
        self._plot_accuracy(x_axis, all_val_accurary, self.lr)  
        
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
        return -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def _evaluate(self, X_val, y_val):
        # Forward pass on entire validation set
        y_hat = self._forward_pass(X_val)
        
        # Compute the loss
        avg_loss = self._loss_function(y_val, y_hat)
        
        # Predicted labels
        predicted_labels = [1 if pred >= 0.5 else 0 for pred in y_hat]
        
        # Compute accuracy
        accuracy = accuracy_score(y_true=y_val, y_pred=predicted_labels)
        
        return avg_loss, accuracy

    
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
        y_pred = self.predict(X_test, self.weights, self.bias)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
    def _sigmoid(self, z):
        """
        Implements the sigmoid function, an activiation function that changes the weighted sum z to a probability.

        Args:
            z (Any): The weighted sum.

        Returns:
            Any: the predicted probability of z (either 0 or 1)
        """
        return 1 / (1 + np.exp(-z))
    
    def _plot_accuracy(self, x_axis, val_accuracy, y_label, lr):
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
        ax.set_ylabel(y_label)
        ax.set_title(f'{y_label} as a Function of Epochs')

        # Save the plot
        plt.savefig(f'moviesense/figures/{y_label.lower()}_epoch_{len(x_axis)}_lr_{lr}.png')

        # plt.show()
        
    def _plot_loss(self, x_axis, train_losses, val_losses, y_label, lr):
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
        ax.set_ylabel(y_label)
        ax.set_title(f'{y_label} as a Function of Epochs')

        # Add legend to distinguish between train/validation
        ax.legend()

        # Save the plot
        plt.savefig(f'moviesense/figures/{y_label.lower()}_epoch_{len(x_axis)}_lr_{lr}.png')
        
        # plt.show()

        
# Running the classifier
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

vectorizer = CountVectorizer()
le = LabelEncoder()

df = pd.read_csv('moviesense/data/cleaned_movie_reviews.csv')
X = vectorizer.fit_transform(df['review'])
y = le.fit_transform(df['sentiment'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train, X_val = X_train.toarray(), X_val.toarray()

print(type(X_train), type(y_train))
print(X_train.shape)

classifer = LogisiticRegression()
classifer.fit(X_train, y_train, X_val, y_val) 
