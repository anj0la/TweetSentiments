import joblib
import pandas as pd
from moviesense.models.logistic_regression import LogisiticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_logisitic_model(lr: float = 0.1, epochs: int = 50, batch_size: int = 64):
    vectorizer = CountVectorizer()
    le = LabelEncoder()

    joblib.dump(vectorizer, 'movie_sense/data/models/logistic_regression/vectorizer.pkl')
    joblib.dump(vectorizer, 'movie_sense/data/models/logistic_regression/le.pkl')

    df = pd.read_csv('moviesense/data/reviews/cleaned_movie_reviews.csv')
    X = vectorizer.fit_transform(df['review'])
    y = le.fit_transform(df['sentiment'].values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train, X_val = X_train.toarray(), X_val.toarray()

    print(type(X_train), type(y_train))
    print(X_train.shape)

    classifer = LogisiticRegression(lr=lr, epochs=epochs, batch_size=batch_size)
    classifer.fit(X_train, y_train, X_val, y_val) 
    accuracy = classifer.evaluate(X_test, y_test)
    print(f'Test Acc: {accuracy * 100:.2f}')
    
train_logisitic_model()