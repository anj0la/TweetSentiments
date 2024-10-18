import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from models.cbow import CBOW
from sklearn.model_selection import train_test_split
from utils.data_preparation import generate_context_target_pairs, one_hot_encode, create_one_hot_vectors

def train(model: CBOW, device: torch.device, vocab: dict[str, int], X: list, y: list, epochs: int = 10, lr: float = 0.01, weight_decay: float = 0.01):
    """
    Trains the model.

    Args:
        model(CBOW): The CBOW model to be trained.
        device(torch.device): The device to perform computations on (CPU or GPU).
        vocab(dict): The vocabulary used for encoding.
        X_train (list): The train context inputs.
        y_train (list): The train target outputs.
        epochs (int, optional): The specified number of iterations to go through the training data. Defaults to 100.
        lr (float, optional): The learning rate to be applied to the SGD. Defaults to 0.01.
        weight_decay(float, optional): L2 normalization. Defaults to 0.01.
    """
    train_total_loss = []
    train_total_accurary = []
    test_total_loss = []
    test_total_accuracy = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        
        train_loss, train_accurary = train_one_epoch(model, device, optimizer, vocab, X_train, y_train)
        print('done training')
        # # Reset total loss for each iteration through training set
        # total_loss = 0

        # # Iterate through training data X
        # for i in range(len(X_train)):
        #     if X_train[i]: # Some training examples are empty -> []
                
        #         # Transform contexts into one hot vectors of type int for embedding layer
        #         ith_context_vect = create_one_hot_vectors(X_train[i], vocab).int().to(device)

        #         # Transform labels into one hot vectors of type int for embedding layer
        #         y_label = one_hot_encode(y_train[i], vocab).to(device)

        #         # Get expected predictions
        #         prediction = model(ith_context_vect)

        #         # Compute cross entropy loss
        #         loss = F.cross_entropy(prediction, y_label)
        #         total_loss += loss.item()

        #         # Backward step
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
                
        #  # Append total loss for training
        train_total_loss.append(train_loss.detach().numpy())
        train_total_accurary.append(train_accurary.detach().numpy())

        # Test the model after each epoch and accumulate losses
        test_loss, test_accuracy = evaluate_one_epoch(model, device, vocab, X_test, y_test)
        print('done evaluating')
        test_total_loss.append(test_loss)
        test_total_accuracy.append(test_accuracy)
        
        # Print train / test metrics
        print(f'\t Epoch: {epoch + 1} out of {epochs}')
        print(f'\t Train Loss: {train_loss:.3f} | Train Acc: {train_accurary * 100:.2f}%')
        print(f'\t Valid Loss: {test_loss:.3f} | Valid Acc: {test_accuracy * 100:.2f}%')
        

    plot_graph(list(range(epochs)), train_total_loss, 'Loss', 'Train Loss')
    plot_graph(list(range(epochs)), test_total_loss, 'Loss', title='Test Loss')
    plot_graph(list(range(epochs)), test_total_accuracy, 'Accuracy', title='Test Accuracy')
    torch.save(model.state_dict(), 'models/cbow_model.pth')
    
def train_one_epoch(model: CBOW, device: torch.device, optimizer: torch.optim.SGD | torch.optim.Adam, vocab: dict[str, int], X_train: list, y_train: list):
    # Reset total loss and correct predictions for each iteration through training set
    model.train()
    total_loss = 0
    correct_predictions = 0

    # Iterate through training data X
    for i in range(len(X_train)):
        # Transform contexts into one hot vectors of type int for embedding layer
        ith_context_vect = create_one_hot_vectors(X_train[i], vocab).int().to(device)

        # Transform labels into one hot vectors of type int for embedding layer
        y_label = one_hot_encode(y_train[i], vocab).to(device)

        # Get expected predictions
        prediction = model(ith_context_vect)

        # Compute cross entropy loss
        loss = F.cross_entropy(prediction, y_label)
        total_loss += loss.item()

        # Backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
            
        # Calculate the number of correct predictions
        predicted_index = torch.argmax(prediction, dim=1)
        true_index = torch.argmax(y_label, dim=1)
        correct_predictions += (predicted_index == true_index).sum().item()
                           
    avg_loss = total_loss / len(X_train)
    accuracy = correct_predictions / len(X_train) * 100
    return avg_loss, accuracy
    
def evaluate_one_epoch(model: CBOW, device: torch.device, vocab: dict[str, int], X_test: list, y_test: list):
    """
    Tests the model on the test dataset.

    Args:
        model (CBOW): The CBOW model to be tested.
        device (torch.device): The device to perform computations on (CPU or GPU).
        vocab (dict): The vocabulary used for encoding.
        X_test (list): The test context inputs.
        y_test (list): The true target outputs.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    
    with torch.no_grad(): # No need to compute gradients
        for i in range(len(X_test)):
            # Transform contexts into one hot vectors of type int for embedding layer
            ith_context_vect = create_one_hot_vectors(X_test[i], vocab).int().to(device)

            # Transform labels into one hot vectors of type int for embedding layer
            y_label = one_hot_encode(y_test[i], vocab).to(device)

            # Get expected predictions
            prediction = model(ith_context_vect)

            # Compute cross entropy loss
            loss = F.cross_entropy(prediction, y_label)
            total_loss += loss.item()  # Accumulate total loss

            # Calculate the number of correct predictions
            predicted_index = torch.argmax(prediction, dim=1)
            true_index = torch.argmax(y_label, dim=1)
            correct_predictions += (predicted_index == true_index).sum().item()

    # Return the average loss and accuracy
    avg_loss = total_loss / len(X_test)
    accuracy = correct_predictions / len(X_test) * 100
    return avg_loss, accuracy

def plot_graph(list_epochs: list, list_total_loss: list, y_label: str, title: str):
    """
    This function plots a graph that visualizes how the loss decreases over the epochs. That is, as the epochs increase, the loss decreases.

    Args:
        list_epochs (list): The list of epochs (iterations).
        list_total_loss (list): The list of total losses per epoch.
    """
    fig, ax = plt.subplots()
    ax.plot(list_epochs, list_total_loss)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.savefig('figure_{}.png'.format(title))
    plt.show()
    
def filter_empty_examples(X: list, y: list):
    filtered_X = []
    filtered_y = []
    
    for context, label in zip(X, y):
        if context:  # Check if the context is not empty
            filtered_X.append(context)
            filtered_y.append(label)
    
    return filtered_X, filtered_y


########### TRAINING THE MODEL ###########

def run():
    # Get the cleaned movie reviews from the dataset
    df = pd.read_csv('moviesense/data/cleaned_movie_reviews.csv')
    
    # Generate training and testing data
    context_target_pairs, vocab = generate_context_target_pairs(df['review'])
    
    unfiltered_X = [data[0] for data in context_target_pairs]
    unfiltered_y = [data[1] for data in context_target_pairs]
    
    X, y = filter_empty_examples(unfiltered_X, unfiltered_y)
    
    # Set up the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cbow = CBOW(vocab_size=len(vocab), hidden_size=200).to(device)
    
    print(device)
    print(cbow)

    # Train the model
    train(model=cbow, device=device, vocab=vocab, X=X, y=y)
    
run()