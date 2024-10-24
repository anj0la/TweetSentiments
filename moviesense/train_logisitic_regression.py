"""
File: train_logistic_regression.py

Author: Anjola Aina
Date Modified: October 22nd, 2024

Description:

This file contains the train_logisitic_model function which is used to train the custom LogisiticRegression model class.
"""
import joblib
import pandas as pd
from models.logistic_regression import LogisiticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_logisitic_model(lr: float = 0.01, epochs: int = 100, batch_size: int = 64, decay_factor: float = 1.0, lr_step: int = 10, reg_lambda: float = 0.01, no_progress_epochs: int = 10) -> None:
    """
    Trains the logisitic regression model.

    Args:
        lr (float, optional): The learning rate. Defaults to 0.01.
        epochs (int, optional): The number of epochs. Defaults to 100.
        batch_size (int, optional): The batch size. Defaults to 64.
        decay_factor (float, optional): The decay factor (how fast the learning rate decreases). Defaults to 1.0.
        lr_step (int, optional): The interval at which the learning rate decreases by the decay factor. Defaults to 10.
        reg_lambda (float, optional): The hyperparameter for L2 regularization. Defaults to 0.01.
        no_progress_epochs (int, optional): The early stopping parameter. Defaults to 10.
    """
    vectorizer_path = 'moviesense/data/models/vectorizer.pkl'
    le_path = 'moviesense/data/models/le.pkl'
    df = pd.read_csv('moviesense/data/reviews/cleaned_movie_reviews.csv')
    vectorizer = CountVectorizer()
    le = LabelEncoder()
    
    # Fit-transform the reviews and sentiments (learns the vocabulary)
    X = vectorizer.fit_transform(df['review'])
    y = le.fit_transform(df['sentiment'].values)
    
    # Save vectorizer and label encoder
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(le, le_path)
        
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Trains (and validates) the model
    classifier = LogisiticRegression(lr=lr, epochs=epochs, batch_size=batch_size, decay_factor=decay_factor, lr_step=lr_step, reg_lambda=reg_lambda, no_progress_epochs=no_progress_epochs)
    classifier.fit(X_train, y_train) 
    
    # Convert to dense array
    X_test = X_test.toarray()
    
    # Evaluate model on test set
    accuracy, precision, recall, f1_score = classifier.evaluate(X_test, y_test)
    
    # Print metrics
    print(f'Test Acc: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')    
    print(f'Recall: {recall * 100:.2f}%')    
    print(f'F1 Score: {f1_score * 100:.2f}%')    
    
train_logisitic_model()