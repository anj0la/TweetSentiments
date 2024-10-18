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
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 100) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.w0 = None # Bias
    
    def _initialize_weights(self) -> None:
        """ 
        Initializes the weights in the binary classifer by assigning fixed values to the weights.
        """
        self.weights = np.array([0.2, -0.3])
        self.w0 = 0.5

    def fit(self, X_train, y_train) -> None:
        """
        Trains the binary classifier on the given training set (Xt) and corresponding probabilities (yt).

        Args:
            X_train (ndarray): The training set.
            y_train (ndarray): The corresponding probabilities for the training set. 
        """
        self._initialize_weights()
        total_loss = 0
        all_train_losses = []
                
        for epoch in range(self.n_iterations):
            total_loss = 0
            
            # SGD (stochastic gradient descent)
            for i in range(len(X_train)):
                
                # Forward pass (make a prediction)
                y_hat = self._forward_pass(X_train[i])
          
                # Calculate the loss (and increment total loss)
                loss = self._binary_cross_entropy(y_train[i], y_hat)
                total_loss += loss
                
                # Backward pass
                grad_w0, grad_w1, grad_w2 = self._backward_pass(X_train[i], y_train[i], y_hat)
                
                # Update weights
                self._update_weights(grad_w0, grad_w1, grad_w2)
                
            # Print loss metrics
            if epoch % 2 == 0:
                print(f'Iteration {epoch} / {self.n_iterations} with total loss: {total_loss / len(X_train)}')
                print(f'Weights : {self.weights}')
            
            all_train_losses.append(total_loss)
            
        # Visualize (and save) plot representing the loss with respect to the epochs
        self._plot_graph(list(range(self.n_iterations)), all_train_losses)  
        
    def _forward_pass(self, x):
        z = self.w0 + np.dot(x, self.weights)
        return self._sigmoid(z)

    def _backward_pass(self, X_train, y_train, y_hat):
        # Backward pass (getting the necessary derivatives for the chain rule)
        dt_loss_yhat = self._deriv_loss_prob(y_train, y_hat)
        dt_yhat_sigmoid = self._deriv_prob_sigmoid(y_hat)
        dt_sigmoid_w0 = 1
        dt_sigmoid_w1 = X_train[0] # x1
        dt_sigmoid_w2 = X_train[1] # x2
                
        # Gradient of loss from respective weight using chain rule
        gradient_loss_w0 = (dt_loss_yhat * dt_yhat_sigmoid * dt_sigmoid_w0)
        gradient_loss_w1 = (dt_loss_yhat * dt_yhat_sigmoid * dt_sigmoid_w1)
        gradient_loss_w2 = (dt_loss_yhat * dt_yhat_sigmoid * dt_sigmoid_w2)
        
        return gradient_loss_w0, gradient_loss_w1, gradient_loss_w2
    
    def _update_weights(self, gradient_loss_w0, gradient_loss_w1, gradient_loss_w2):
        self.w0 -= self.learning_rate * gradient_loss_w0
        self.weights[0] -= self.learning_rate * gradient_loss_w1
        self.weights[1] -= self.learning_rate * gradient_loss_w2
    
    def predict(self, X):
        """
        Predicts the probability (output either 0 or 1) for a given input X, by using the sigmoid function.
        As the sigmoid function may give a decimal value, we use np.round so that values over 0.5 (inclusive) are rounded up to 1,
        and values less than 0.5 (exclusive) are rounded down to 0.

        Args:
            X (ndarray): the input to make a prediction on

        Returns:
            int: the predicted probability of the input (either 0 or 1)
        """
        z = np.dot(X, self.weights) + self.w0
        return np.round(self._sigmoid(z))
    
    def evaluate(self, X):
        pass
    
    def _sigmoid(self, z):
        """
        Implements the sigmoid function, an activiation function that changes the weighted sum z to a probability.

        Args:
            z (Any): the weighted sum

        Returns:
            Any: the predicted probability of z (either 0 or 1)
        """
        return 1 / (1 + np.exp(-z))
    
    def _binary_cross_entropy(self, y, y_hat):
        """
        Implements the binary cross entropy algorithm, a loss function used for logistic regression models to penalize misclassification.

        Args:
            y (Any): the true probability (ground truth)
            y_hat (Any): the predicted label

        Returns:
            Any: the loss of the model
        """
        return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def _deriv_loss_prob(self, y, y_hat):
        """
        Calculates the derivative of the loss with respect to the predicted probability y_hat.

        Args:
            y (Any): the true probability
            y_hat (Any): the predicted probability

        Returns:
            Any: the derivative of the loss with respect to the predicted probability
        """
        return -((y / y_hat) - ((1 - y) / (1 - y_hat)))
    
    def _deriv_prob_sigmoid(self, y_hat):
        """
        Calculates the derivative of the predicted probability y_hat with respect to the input of the sigmoid function z.

        Args:
            y_hat (Any): the predicted probability

        Returns:
            Any: the derivative of the predicted probability y_hat with respect to the input of the sigmoid function
        """
        return y_hat * (1 - y_hat)
    
    def _plot_graph(self, list_epochs, list_total_loss):
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
        ax.set_title('Binary Cross Entropy Loss Function as a Function of Epochs')
        plt.savefig('logistic_regression_loss.png')
        plt.show()
        
# Running the classifier
classifer = LogisiticRegression()
classifer.fit(X_train, y_train) 

y_pred = []
for i in range(len(X_test)):
    y_pred.append(classifer.predict(X_test[i]))

print(f'Accuracy score: {accuracy_score(y_true=y_test, y_pred=y_pred)}')