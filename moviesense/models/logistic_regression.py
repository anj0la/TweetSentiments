import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

num_samples_per_class = 100

np.random.seed(0)
X1 = np.random.randn(num_samples_per_class, 2) + np.array([2, 2])
X2 = np.random.randn(num_samples_per_class, 2) + np.array([-2, -2])
X = np.vstack([X1, X2])
y = np.array([0] * num_samples_per_class + [1] * num_samples_per_class)

shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class LogisiticRegression:
    """
    A class to represent the implementation of a logistic regression with a sigmoid activation function.
    
    Attributes:
        - learning_rate - the learning rate of the classifier
        - n_iterations - the number of iterations to go through all of the training examples ( known as epochs) 
        - weights - the parameters of the model, adjusted to get the desired probability
        - w0 - an extra parameter, adjusted to get the desired probability
        
    Public Methods:
        - fit(inputs, actual_labels) -> trains the perceptron on a training set and the corresponding labels that each training example should map to
        - predict(input) -> given a sentence, returns the perceptron's prediction (positive or negative)
    """
    
    def __init__(self, learning_rate=0.1, n_iterations=100):
        """
        Constructs all of the necessary attributes for the Binary Classifier object.
        
        Args:
            - learning_rate - the learning rate to be applied to the model. If no value is specified, the default learning rate is 0.1
            - n_iterations - the number of iterations (epochs) that the model will go through to learn the labels. If no value is specified, the default value is 100
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.w0 = None # Bias
    
    def _initialize_weights(self):
        """ 
        Initializes the weights in the binary classifer by assigning fixed values to the weights.
        """
        self.weights = np.array([0.2, -0.3])
        self.w0 = 0.5

    def fit(self, Xt, yt):
        """
        Trains the binary classifier on the given training set (Xt) and corresponding probabilities (yt).

        Args:
            Xt (ndarray): the training set 
            yt (ndarray): the corresponding probabilities for the training set 
        """
        self._initialize_weights()
        total_loss = 0
        list_total_loss = []
        list_epochs = []
                
        for epoch in range(self.n_iterations):
            # Start the loop
            total_loss = 0
            # SGD (stochastic gradient descent)
            for i in range(len(Xt)):
                
                # Forward pass (make a prediction)
                #z = self.w0 + np.dot(Xt[i], self.weights)
                #y_hat = self._sigmoid(z)
                y_hat = self._forward_pass(Xt[i])
          
                # Calculate the loss (and increment total loss)
                loss = self._binary_cross_entropy(yt[i], y_hat)
                total_loss += loss
                
                grad_w0, grad_w1, grad_w2 = self._backward_pass(Xt[i], yt[i], y_hat)
                
                # # backward pass (getting the necessary derivatives for the chain rule)
                # dt_loss_yhat = self._deriv_loss_prob(yt[i], y_hat)
                # dt_yhat_sigmoid = self._deriv_prob_sigmoid(y_hat)
                # dt_sigmoid_w0 = 1
                # dt_sigmoid_w1 = Xt[i][0] # x1
                # dt_sigmoid_w2 = Xt[i][1] # x2
                
                # # gradient of loss from respective weight using chain rule
                # gradient_loss_w0 = (dt_loss_yhat * dt_yhat_sigmoid * dt_sigmoid_w0)
                # gradient_loss_w1 = (dt_loss_yhat * dt_yhat_sigmoid * dt_sigmoid_w1)
                # gradient_loss_w2 = (dt_loss_yhat * dt_yhat_sigmoid * dt_sigmoid_w2)
                
                self._update_weights(grad_w0, grad_w1, grad_w2)
                
                # updating weights
                # self.w0 -= self.learning_rate * gradient_loss_w0
                # self.weights[0] -= self.learning_rate * gradient_loss_w1
                # self.weights[1] -= self.learning_rate * gradient_loss_w2
                
            if epoch % 2 == 0:
                print(f'Iteration {epoch} / {self.n_iterations} with total loss: {total_loss / len(Xt)}')
                print(f'Weights : {self.weights}')
            
            # to plot the loss as a function of the epochs (visualizing the loss)
            list_total_loss.append(total_loss)
            list_epochs.append(epoch)
            
        # after reaching the max number of iterations (= epochs), visualize the loss with respect to the epochs with a plot
        self._plot_graph(list_epochs, list_total_loss)  
        
    def _forward_pass(self, x):
        z = self.w0 + np.dot(x, self.weights)
        return self._sigmoid(z)

    def _backward_pass(self, Xt, yt, y_hat):
        # Backward pass (getting the necessary derivatives for the chain rule)
        dt_loss_yhat = self._deriv_loss_prob(yt, y_hat)
        dt_yhat_sigmoid = self._deriv_prob_sigmoid(y_hat)
        dt_sigmoid_w0 = 1
        dt_sigmoid_w1 = Xt[0] # x1
        dt_sigmoid_w2 = Xt[1] # x2
                
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
    
    def predictions(self, X):
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
        plt.show()
        
# Running the classifier
classifer = LogisiticRegression()
classifer.fit(X_train, y_train) 

y_pred = []
for i in range(len(X_test)):
    y_pred.append(classifer.predict(X_test[i]))

print(f'Accuracy score: {accuracy_score(y_true=y_test, y_pred=y_pred)}')