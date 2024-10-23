import joblib
import numpy as np
import pandas as pd
from models.logistic_regression import LogisiticRegression
from utils.preprocess import clean_review

def run_model(model_name: str) -> None:
    vectorizer_path = 'moviesense/data/models/vectorizer.pkl'
    le_path = 'moviesense/data/models/le.pkl'
    sentence = 'I hated the movie, it was so bad'
    
    if model_name == 'LR':
        run_logistic_regression(vectorizer_path, le_path, sentence)
    
def run_logistic_regression(vect_path: str, le_path: str, sentence: str) -> None:
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
    prediction = np.round(classifier.predict(vectorized_sentence))    
    label = le.inverse_transform(prediction.astype(int).flatten())[0]
    
    # Print the results
    print(f'\tSentence: {sentence} \n\tPrediction: {label}')
    
    
if __name__ == '__main__':
    run_model(model_name='LR')
