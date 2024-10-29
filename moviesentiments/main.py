"""
File: main.py

Author: Anjola Aina
Date Modified: October 27th, 2024

Description:

This file is used to run all of the trained models.
"""
import joblib
import pandas as pd
import torch
from models.logistic_regression import LogisticRegression
from models.mlp import MLP
from models.rnn import RNN, GRU, LSTM
from train_rnn import initialize_model
from utils.preprocess import clean_review, word_to_idx

def load_model(vocab: dict[str, int], model_name: str) -> LogisticRegression | MLP | RNN | GRU | LSTM:
    if model_name == 'LR':
        model = LogisticRegression()
        model.load_model()
    elif model_name == 'MLP':
        model = MLP(len(vocab) - 2)
        model.load_state_dict(torch.load('moviesense/data/models/mlp/mlp_saved_state.pt', weights_only=True))
        model.eval()
    elif model_name in ['RNN', 'GRU', 'LSTM', 'BiRNN', 'BiGRU', 'BiLSTM']:
        model = initialize_model(model_name=model_name, vocab_size=len(vocab), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), bidirectional=('Bi' in model_name))
        model.load_state_dict(torch.load(f'moviesense/data/models/rnn/{model_name.lower()}_saved_state.pt', weights_only=True))
        model.eval()
    else:
        raise ValueError(f'Invalid model name "{model_name}". Expected one of LR, MLP, RNN, GRU, LSTM, BiRNN, BiGRU, or BiLSTM.')
        
    return model

def make_prediction(sentence: str, model_name: str, model: LogisticRegression | MLP | RNN | GRU | LSTM) -> int:
    if model_name == 'LR':
        prediction = model.predict(sentence)
        prediction = prediction.astype(int).flatten()
    elif model_name == 'MLP':
        input = torch.tensor(sentence, dtype=torch.float)
        prediction = torch.sigmoid(model(input))
        prediction = torch.round(prediction).detach().numpy().astype(int)[0]
    else: # RNN-like model
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
    else: # RNN-like model (exception is caught when making the model)
        encoded_sentence = word_to_idx(' '.join(cleaned_sentence.tolist()), vocab, len(vocab) + 1)
        prediction = make_prediction(encoded_sentence, model_name, model)
    
    label = le.inverse_transform(prediction)

    # Print the results
    print(f'\tSentence: {sentence} \n\tPrediction: {label}')
    
if __name__ == '__main__':
    #pass
    run_model(sentence='I loved the movie, it was so good', model_name='RNN')
