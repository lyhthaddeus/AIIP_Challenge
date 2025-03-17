# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from model import train_xgb_model
from preprocess import clean_data, load_data, preprocess_features  # Import the train_xgb_model function

def load_and_preprocess_data():
    data = load_data()
    df = data['farm_data']

    df = clean_data(df)
    df = preprocess_features(df)

    X = df.drop(columns=['TEMPERATURE SENSOR (°C)'])
    y = df['TEMPERATURE SENSOR (°C)']
    X = X.loc[y.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    trained_model = train_xgb_model(X_train, y_train, X_test, y_test)

    return trained_model

if __name__ == "__main__":
    trained_model = train_and_evaluate()