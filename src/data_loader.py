import pandas as pd

def load_dataset(file_path):
    """Loads dataset from a given file path"""
    return pd.read_csv(file_path)

def check_missing_values(df):
    """Returns missing value count per column"""
    return df.isnull().sum()

def preprocess_data(df):
    """Basic preprocessing like removing duplicates, standardizing column names"""
    #df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df
