"""
File: main.py

Author: Anjola Aina
Date Modified: October 24th, 2024

Description:

This file is used to run all of the trained models.
"""
import joblib
import pandas as pd
import torch
from models.logistic_regression import LogisiticRegression
from models.mlp import MLP
from models.rnn import RNN
from sklearn.feature_extraction.text import CountVectorizer

from utils.preprocess import clean_review, text_to_sequence

def load_model(vocab: dict[str, int], model_name: str):
    if model_name == 'LR':
        # Load the trained weights and bias for the model
        model = LogisiticRegression()
        model.load_model()
    elif model_name == 'MLP':
        # Load MLP from saved state and set to eval mode
        model = MLP(len(vocab) - 2) # Need to retrain MLP on padded and unk tokens
        model.load_state_dict(torch.load('moviesense/data/models/mlp/mlp_saved_state.pt', weights_only=True))
        model.eval()
    elif model_name == 'RNN':
        model = RNN(len(vocab))
        model.load_state_dict(torch.load('moviesense/data/models/rnn/rnn_saved_state.pt', weights_only=True))
        model.eval()
        
    return model

def make_prediction(sentence: str, model_name: str, model: LogisiticRegression | MLP | RNN) -> int:
    if model_name == 'LR':
        prediction = model.predict(sentence)
        prediction = prediction.astype(int).flatten()
    elif model_name == 'MLP':
        input = torch.tensor(sentence, dtype=torch.float)
        prediction = torch.sigmoid(model(input))
        prediction = torch.round(prediction).detach().numpy().astype(int)[0]
    else: # RNN-like
        input = torch.tensor(sentence).unsqueeze(1).T  # Reshaping in form of batch, number of words
        lengths = torch.tensor([len(sentence)], dtype=torch.long)
        prediction = torch.sigmoid(model(input, lengths))
        prediction = torch.round(prediction).detach().numpy().astype(int)[0]
        
    return prediction

def run_model(sentence: str, model_name: str = 'LR') -> None:
    """
    Makes a prediction with the specificed model.

    Args:
        model_name (str): The model used to make a prediction. Defaults to LR.
    """
    vectorizer_path = 'moviesense/data/models/vectorizer.pkl'
    le_path = 'moviesense/data/models/le.pkl'
    
    # Load the trained vectorizer and label encoder    
    vectorizer = joblib.load(vectorizer_path)
    le = joblib.load(le_path)
    
    # Set the vocabulary (need pad and unk token to get correct size of vocabulary)
    vocab_size = len(vectorizer.vocabulary_)
    vocab = {'<pad>': 0, '<unk>': vocab_size + 1, **vectorizer.vocabulary_}
    
    # Convert sentence to Dataframe for easier processing
    df = pd.DataFrame({'review': [sentence]})
    
    # Transform the review into a suitable input
    cleaned_sentence = clean_review(df)
    
    # Get the selected model
    model = load_model(vocab, model_name)
    
    # Make a prediction
    if model_name == 'LR' or model_name == 'MLP':
        vectorized_sentence = vectorizer.transform(cleaned_sentence).toarray()
        prediction = make_prediction(vectorized_sentence, model_name, model)
    else: # RNN-like model
        encoded_sentence = text_to_sequence(' '.join(cleaned_sentence.tolist()), vocab, len(vocab) + 1)
        prediction = make_prediction(encoded_sentence, model_name, model)
    
    label = le.inverse_transform(prediction)
    
    """ # Logisitic Regression
    if model_name == 'LR':
        # Load the trained weights and bias for the model
        model = LogisiticRegression()
        model.load_model()
    elif model_name == 'MLP':
        # Load MLP from saved state and set to eval mode
        model = MLP(len(vectorizer.vocabulary_))
        model.load_state_dict(torch.load('moviesense/data/models/mlp/mlp_saved_state.pt', weights_only=True))
        model.eval() """
    
    # # Make a prediction
    # if model_name == 'LR':
    #     prediction = model.predict(vectorized_sentence)
    #     prediction = prediction.astype(int).flatten()
    #     label = le.inverse_transform(prediction)
    # elif model_name == 'MLP':
    #     prediction = torch.sigmoid(model(torch.tensor(vectorized_sentence, dtype=torch.float)))
    #     prediction = torch.round(prediction).detach().numpy().astype(int)[0]
    #     label = le.inverse_transform(prediction)
    
    # Print the results
    print(f'\tSentence: {sentence} \n\tPrediction: {label}')
    
if __name__ == '__main__':
    #pass
    run_model(sentence='I loved the movie, it was so good', model_name='RNN')
