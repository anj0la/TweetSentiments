"""
File: logistic_regression.py

Author: Anjola Aina
Date Modified: October 27th, 2024

Description:

This file contains the LogisticRegression class which is used to implement a binary classifier with a sigmoid activation function.
Source for early stopping: https://medium.com/@juanc.olamendy/understanding-early-stopping-a-key-to-preventing-overfitting-in-machine-learning-17554fc321ff
"""
import numpy as np
from utils.plot_graphs import plot_loss, plot_accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, lr: float = 0.1, epochs: int = 10, batch_size: int = 64, decay_factor: float = 0.1, lr_step: int = 10, reg_lambda: float = 0.0, no_progress_epochs: int = 10) -> None:
        """
        A class representing the implementation of a logistic regression with a sigmoid activation function.

        Args:
            lr (float, optional): The learning rate. Defaults to 0.01.
            epochs (int, optional): The number of epochs. Defaults to 100.
            batch_size (int, optional): The batch size. Defaults to 64.
            decay_factor (float, optional): The decay factor (how fast the learning rate decreases). Defaults to 1.0.
            lr_step (int, optional): The interval at which the learning rate decreases by the decay factor. Defaults to 10.
            reg_lambda (float, optional): The hyperparameter for L2 regularization. Defaults to 0.01.
            no_progress_epochs (int, optional): The early stopping parameter. Defaults to 10.
        """
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None # Bias
        self.batch_size = batch_size
        self.decay_factor = decay_factor
        self.lr_step = lr_step
        self.reg_lambda = reg_lambda
        self.no_progress_epochs = no_progress_epochs
    
    def _initialize_weights(self, n_features: int) -> None:
        """ 
        Initializes the weights in the binary classifer by assigning fixed values to the weights.
        
        Args:
            n_features(int): The number of features.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0.5
        
    def _forward_pass(self, x: np.ndarray) -> np.ndarray:
        """
        Implements the forward pass for a logistic regression model.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The predicted probability from the activation function.
        """
        z = self.bias + np.dot(x, self.weights)
        return self._sigmoid(z)
        
    def _update_weights(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> None:
        """
        Updates the weights and bias.

        Args:
            X (np.ndarray): The training set.
            y (np.ndarray): The corresponding labels for the training set.
            y_hat (np.ndarray): The predicted labels from the training set.
        """
        m = X.shape[0]
        d_weight = (1 / m) * np.dot(X.T, (y_hat - y))
        d_bias = (1 / m) * np.sum(y_hat - y)
        self.weights -= self.lr * d_weight
        self.bias -= self.lr * d_bias
            
    def _loss_function(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Implements the total cross entropy loss function.

        Args:
            y (np.ndarray): The true labels.
            y_hat (np.ndarray): The predicted labels.

        Returns:
            float: The total loss from the predicted labels.
        """
        # Clip y_hat to avoid log(0)
        y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
        m = y.shape[0]
        
        cross_entropy_loss = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        l2_reg = (self.reg_lambda / (2 * m)) * np.sum(np.square(self.weights))
        return cross_entropy_loss + l2_reg
       
    def _evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
        """
        Evaluates the model on the validation set.

        Args:
            X_val (np.ndarray): The validation set.
            y_val (np.ndarray): The corresponding labels for the validation set.

        Returns:
            tuple[float, float]: A tuple containing the loss and accuracy score for the validation set.
        """
        # Forward pass on entire validation set
        y_hat = self._forward_pass(X_val)
        
        # Compute the loss
        loss = self._loss_function(y_val, y_hat)
        
        # Predicted labels
        predicted_labels = [1 if pred >= 0.5 else 0 for pred in y_hat]
        
        # Compute accuracy
        accuracy = accuracy_score(y_true=y_val, y_pred=predicted_labels)
        
        return loss, accuracy
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Implements the sigmoid function, an activiation function that changes the weighted sum z to a probability.

        Args:
            z (np.ndarray): The weighted sum.

        Returns:
            np.ndarray: The predicted probability of z.
        """
        return 1 / (1 + np.exp(-z))
        
    def _save_model(self) -> None:
        """
        Saves the trained weights and bias.
        """
        np.save('moviesense/data/models/logistic_regression/weights.npy', self.weights)
        np.save('moviesense/data/models/logistic_regression/bias.npy', np.array(self.bias))
        
    def load_model(self) -> None:
        """
        Loads the model by setting the weights and bias to be its trained values.
        """
        self.weights = np.load('moviesense/data/models/logistic_regression/weights.npy')
        self.bias = np.load('moviesense/data/models/logistic_regression/bias.npy')

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the model.

        Args:
            X_train (np.ndarray): The training set.
            y_train (np.ndarray): The corresponding labels for the testing set.
        """
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Convert to dense arrays to work with model
        X_train, X_val = X_train.toarray(), X_val.toarray()
        
        self._initialize_weights(X_train.shape[1])
        all_train_losses = []
        all_val_losses = []
        all_train_accuracy = []
        all_val_accuracy = []
        num_batches = X_train.shape[0] // self.batch_size
        no_progress_count = 0
        min_val_loss = float('inf')
        
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
            
            # Save the weights and bias to be used for predictions
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                self._save_model()
                no_progress_count = 0
            else:
                no_progress_count += 1
                
            # Early stopping condition
            if no_progress_count >= self.no_progress_epochs:
                self.epochs = epoch # Done so we only display results up until the epoch where we broke out of the loop
                print(f'Stopping early at epoch: {epoch + 1}')
                break

            # Append train, val losses and accurary
            all_train_losses.append(avg_loss)
            all_train_accuracy.append(train_accuracy)
            all_val_losses.append(val_loss)
            all_val_accuracy.append(round(val_accuracy, 2))
            
            # Print train and val metrics
            print(f'\t Epoch: {epoch + 1} out of {self.epochs}')
            print(f'\t Train Loss: {avg_loss:.3f} | Train Acc: {train_accuracy * 100:.2f}%')
            print(f'\t Valid Loss: {val_loss:.3f} | Valid Acc: {val_accuracy * 100:.2f}%')
            
        # Visualize (and save) plots
        x_axis = list(range(1, self.epochs + 1))
        plot_loss(x_axis=x_axis, train_losses=all_train_losses, val_losses=all_val_losses, figure_path=f'moviesense/figures/logistic_regression/loss_epoch_{len(x_axis)}_lr_{self.lr}.png')
        plot_accuracy(x_axis=x_axis, train_accuracy=all_train_accuracy,val_accuracy= all_val_accuracy, figure_path=f'moviesense/figures/logistic_regression/accuracy_epoch_{len(x_axis)}_lr_{self.lr}.png')

    def predict(self, X: np.ndarray) -> int:
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
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluates the trained model on the testing set.

        Args:
            X_test (np.ndarray): The testing set.
            y_test (np.ndarray): The corresponding labels for the testing set.

        Returns:
            float: The accuracy score.
        """
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        return accuracy, precision, recall, f1
