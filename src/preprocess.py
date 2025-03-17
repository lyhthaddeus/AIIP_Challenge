import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import os

DB_PATH = os.getenv("DB_PATH", "agri.db")

def load_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    table_names = [t[0] for t in tables]
    dataframes = {t: pd.read_sql_query(f"SELECT * FROM {t}", conn) for t in table_names}
    conn.close()
    return dataframes

def clean_data(df):
    df.columns = df.columns.str.upper()
    df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x) 
    df.drop_duplicates(inplace=True)  # Remove duplicate rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if 'SYSTEM LOCATION CODE' in df.columns:
        df = df.groupby('SYSTEM LOCATION CODE').apply(lambda group: group.interpolate())
    
    df.dropna(inplace=True)    
    
    return df

def preprocess_features(df):
    """Perform feature engineering: normalize numerical features and encode categorical variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Normalize numerical features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df

if __name__ == "__main__":
    data = load_data()
    for table, df in data.items():
        df = clean_data(df)
        df = preprocess_features(df)
        print(f"Processed {table} with shape {df.shape}")
        print("Missing Values:")
        print(df.isnull().sum())
        print("\nDuplicate Rows (Exact Match):", df.duplicated().sum())