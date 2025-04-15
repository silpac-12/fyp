import numpy as np
from sklearn.impute import SimpleImputer
from fancyimpute import IterativeImputer  # MICE implementation
import pandas as pd

def identify_column_types(df):
    """Strictly separates numerical and categorical columns while removing datetime."""

    # Ensure no datetime columns exist
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    df = df.drop(columns=datetime_cols, errors="ignore")  # Drop datetime columns

    # Now classify numerical vs categorical
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return df, numerical_cols, categorical_cols, datetime_cols

def encode_categorical(df):
    """Encodes only `object` type categorical columns while preserving NaN values."""
    df = df.copy()
    mappings = {}

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        df[col] = df[col].astype("category")
        mapping = dict(enumerate(df[col].cat.categories))  # Create category mapping
        reverse_mapping = {v: k for k, v in mapping.items()}  # Reverse for encoding
        mappings[col] = reverse_mapping  # Save reverse mapping for decoding

        df[col] = df[col].map(reverse_mapping)  # Encode categories to integers
        df[col] = df[col].astype("Int64")  # Preserve NaN with nullable integer

    return df, mappings

def decode_categorical(df, mappings):
    df = df.copy()

    for col, mapping in mappings.items():
        if col in df.columns:
            # Ensure column is integer type before decoding
            #df[col] = df[col].astype("Int64")
            # Decode
            df[col] = df[col].map({v: k for k, v in mapping.items()})

    return df


def apply_imputation(df, method, mappings):
    import numpy as np
    from sklearn.impute import SimpleImputer, IterativeImputer

    df = df.copy().replace({pd.NA: np.nan})  # Replace pd.NA with np.nan

    # Select imputation method
    if method == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif method == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    elif method == "❗ MICE (Recommended)":
        imputer = IterativeImputer(max_iter=10, random_state=42)
    else:
        raise ValueError("Unsupported imputation method.")

    # Impute data
    df[df.columns] = imputer.fit_transform(df)

    # ✅ Round and convert only categorical columns to Int64
    for col in mappings:
        if col in df.columns:
            df[col] = df[col].round().astype("Int64")

    return df
