import joblib
import pandas as pd
from models.logistic_regression import LogisiticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_logisitic_model(lr: float = 0.01, epochs: int = 75, batch_size: int = 64, decay_factor: float = 1.0, lr_step: int = 10, reg_lambda: float = 0.01):
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
    joblib.dump(vectorizer, le_path)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training (and validating) the model
    classifer = LogisiticRegression(lr=lr, epochs=epochs, batch_size=batch_size, decay_factor=decay_factor, lr_step=lr_step, reg_lambda=reg_lambda)
    classifer.fit(X_train, y_train) 
    
    # Convert to dense array
    X_test = X_test.toarray()
    
    # Evaluating model on test set
    accuracy = classifer.evaluate(X_test, y_test)
    print(f'Test Acc: {accuracy * 100:.2f}%')
    
train_logisitic_model()