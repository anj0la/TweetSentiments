"""
File: main.py

Author: Anjola Aina
Date Modified: October 22nd, 2024

Description:

This file is used to run all of the trained models.
"""
import joblib
import pandas as pd
from models.logistic_regression import LogisiticRegression
from utils.preprocess import clean_review

def run_model(model_name: str = 'LR') -> None:
    """
    Runs the specified model.

    Args:
        model_name (str): The model to run. Defaults to LR.
    """
    vectorizer_path = 'moviesense/data/models/vectorizer.pkl'
    le_path = 'moviesense/data/models/le.pkl'
    sentence = 'I hated the movie, it was so bad'
    
    if model_name == 'LR':
        run_logistic_regression(vectorizer_path, le_path, sentence)
    
def run_logistic_regression(vect_path: str, le_path: str, sentence: str) -> None:
    """
    Runs the trained logisitic regression to predict the sentiment of the given sentence.

    Args:
        vect_path (str): The path to the trained vectorizer.
        le_path (str): The path to the trained label encoder.
        sentence (str): The sentence to make a prediction on.
    """
    # Load the trained weights and bias from the model
    classifier = LogisiticRegression()
    classifier.load_model()
    
    # Load the trained vectorizer and label encoder    
    vectorizer = joblib.load(vect_path)
    le = joblib.load(le_path)
        
    # Convert sentence to Dataframe for easier processing
    df = pd.DataFrame({'review': [sentence]})
    
    # Transform the review into a suitable input
    cleaned_sentence = clean_review(df)
    vectorized_sentence = vectorizer.transform(cleaned_sentence).toarray()
    
    # Make a prediction
    prediction = classifier.predict(vectorized_sentence)
    label = le.inverse_transform(prediction.astype(int).flatten())[0]
    
    # Print the results
    print(f'\tSentence: {sentence} \n\tPrediction: {label}')
    
if __name__ == '__main__':
    pass
    # run_model(model_name='LR')
