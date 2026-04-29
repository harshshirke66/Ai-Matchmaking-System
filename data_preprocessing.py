import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils import load_data
import os

def preprocess_data(input_path, output_path):
    print('--- Starting Data Preprocessing ---')
    df = load_data(input_path)
    print(f'Initial dataset shape: {df.shape}')
    missing_vals = df.isnull().sum()
    if missing_vals.any():
        print('Missing values detected. Handling...')
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    else:
        print('No missing values found.')
    positive_outcomes = ['Relationship Formed', 'Date Happened', 'Mutual Match', 'Instant Match']
    df['is_match'] = df['match_outcome'].apply(lambda x: 1 if x in positive_outcomes else 0)
    print(f"Target distribution:\n{df['is_match'].value_counts(normalize=True)}")
    df['num_interests'] = df['interest_tags'].apply(lambda x: len(str(x).split(',')))
    cols_to_drop = ['match_outcome', 'interest_tags']
    df = df.drop(columns=cols_to_drop)
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f'Encoding categorical columns: {list(categorical_cols)}')
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    df.to_csv(output_path, index=False)
    print(f'Preprocessed data saved to {output_path}')
    print('--- Preprocessing Complete ---')
    return df
if __name__ == '__main__':
    input_file = 'dating_dataset.csv'
    output_file = 'cleaned_dating_data.csv'
    if os.path.exists(input_file):
        preprocess_data(input_file, output_file)
    else:
        print(f'Error: {input_file} not found. Please ensure the dataset exists.')