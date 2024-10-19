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
    def __init__(self, lr: float = 0.1, epochs: int = 100) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None # Bias
    
    def _initialize_weights(self, n_features) -> None:
        """ 
        Initializes the weights in the binary classifer by assigning fixed values to the weights.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0.5

    def fit(self, X_train, y_train) -> None:
        """
        Trains the binary classifier on the given training set (Xt) and corresponding probabilities (yt).

        Args:
            X_train (ndarray): The training set.
            y_train (ndarray): The corresponding probabilities for the training set. 
        """
        self._initialize_weights(X_train.shape[1])
        all_train_losses = []
        # best_loss = []
        print('Starting training...')
                
        for epoch in range(self.epochs):
            # Forward pass (make a prediction)
            y_hat = self._forward_pass(X_train)
            
            print('WE MADE A FORWARD PASS!')
            
            #print(f'type of y_train: {type(y_train)}, type of y_hat: {type(y_hat)}')
            #print(f'y_train[:5]: {y_train[:5]}, y_hat[:5]: {y_hat[:5]}')
            #print(f'y_train type: {y_train.dtype}, y_hat type: {y_hat.dtype}')

            # Calculate the loss
            loss = self._loss_function(y_train, y_hat)
            
            print('WE CALCLUATED THE LOSS!')
            
            # Backward pass and update weights
            self._update_weights(X_train, y_train, y_hat)
            
            print('WE UPDATED THE WEIGHTS!')
                
            # Print loss metrics
            print(f'Epoch {epoch + 1} / {self.epochs}\nCurrent loss: {loss}')
            # print(f'Weights : {self.weights}')
            
            all_train_losses.append(loss)
            
        # Visualize (and save) plot representing the loss with respect to the epochs
        self._plot_graph(list(range(1, self.epochs + 1)), all_train_losses, self.lr)  
        
    def _forward_pass(self, x):
        z = self.bias + np.dot(x, self.weights)
        return self._sigmoid(z)
        
    def _update_weights(self, X, y, y_hat):
        m = X.shape[0]
        d_weight = (1 / m) * np.dot(X.T, (y_hat - y))
        d_bias = (1 / m) * np.sum(y_hat - y)
        self.weights -= self.lr * d_weight
        self.bias -= self.lr * d_bias
            
    def _loss_function(self, y_train, y_hat):
        m = y_train.shape[0]
        return -(1 / m) * np.sum(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat))
    
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
    
    def _plot_graph(self, list_epochs, list_total_loss, lr):
        """
        This function plots a graph that visualizes how the loss decreases over the epochs. That is, as the epochs increase, the loss decreases.

        Args:
            list_epochs (list): all the epochs (iterations)
            list_total_loss (list): all the total losses per epoch
        """
        fig, ax = plt.subplots()
        ax.plot(list_epochs, list_total_loss) 
        ax.set_xlabel('Number of epochs')
        ax.set_ylabel('Total loss')
        ax.set_title('Loss as a Function of Epochs')
        plt.savefig(f'moviesense/figures/loss_epoch_{len(list_epochs)}_lr_{lr}.png')
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = X_train.toarray()

print(type(X_train), type(y_train))
print(X_train.shape)

classifer = LogisiticRegression()
classifer.fit(X_train, y_train) 
