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
from utils.preprocess import clean_review

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
    
    # Convert sentence to Dataframe for easier processing
    df = pd.DataFrame({'review': [sentence]})
    
    # Transform the review into a suitable input
    cleaned_sentence = clean_review(df)
    vectorized_sentence = vectorizer.transform(cleaned_sentence).toarray()
    
    # Logisitic Regression
    if model_name == 'LR':
        # Load the trained weights and bias for the model
        model = LogisiticRegression()
        model.load_model()
    elif model_name == 'MLP':
        # Load MLP from saved state and set to eval mode
        model = MLP(len(vectorizer.vocabulary_))
        model.load_state_dict(torch.load('moviesense/data/models/mlp/mlp_saved_state.pt', weights_only=True))
        model.eval()
    
    # Make a prediction
    if model_name == 'LR':
        prediction = model.predict(vectorized_sentence)
        prediction = prediction.astype(int).flatten()
        label = le.inverse_transform(prediction)
    else:
        prediction = torch.sigmoid(model(torch.tensor(vectorized_sentence, dtype=torch.float)))
        prediction = torch.round(prediction).detach().numpy().astype(int)[0]
        label = le.inverse_transform(prediction)
    
    # Print the results
    print(f'\tSentence: {sentence} \n\tPrediction: {label}')
    
if __name__ == '__main__':
    pass
    # run_model(sentence='I hated the movie, it was so bad', model_name='LR')
