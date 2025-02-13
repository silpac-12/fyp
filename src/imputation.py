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
    """Encodes only `object` type categorical columns while leaving missing values (NaN) unchanged."""

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()  # Only process `object` columns
    mappings = {}

    for col in categorical_cols:
        df[col] = df[col].astype("category")  # Convert to categorical
        mapping = dict(enumerate(df[col].cat.categories))  # Create category mapping
        df[col] = df[col].cat.codes  # Convert to numerical codes
        df[col].replace(-1, pd.NA, inplace=True)  # Keep NaN values unchanged
        mappings[col] = mapping  # Store mapping for decoding

    return df, mappings


def decode_categorical(df, mappings):
    """Decodes categorical columns back to their original labels, ensuring correct dtype."""

    for col, mapping in mappings.items():
        df[col] = df[col].round().astype(int)  # Ensure valid integer labels before mapping
        df[col] = df[col].map(mapping)  # Convert numeric codes back to original categories
        df[col] = df[col].astype("string")  # Ensure correct dtype for Arrow serialization

    return df


def apply_imputation(df, method="mean", mappings=None):
    """Encodes categorical data, imputes missing values, and correctly decodes categorical values."""

    # Encode categorical columns before imputation
    df_encoded, mappings = encode_categorical(df)

    # Convert all missing values (`pd.NA`) to `np.nan`
    df_encoded = df_encoded.replace({pd.NA: np.nan})

    # Select imputation method (applies to both numerical and encoded categorical data)
    if method == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif method == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    elif method == "mice":
        imputer = IterativeImputer(max_iter=50, random_state=42)  # MICE Imputation
    else:
        raise ValueError("Unsupported imputation method. Choose 'mean', 'zero', or 'mice'.")

    # Apply imputation to all columns
    df_encoded[df_encoded.columns] = imputer.fit_transform(df_encoded)

    # Ensure categorical values are rounded before decoding
    categorical_cols = mappings.keys()
    for col in categorical_cols:
        df_encoded[col] = np.round(df_encoded[col]).astype("Int64")  # Ensure valid category labels

    # Decode categorical columns back to original values
    df_final = decode_categorical(df_encoded, mappings)

    return df_final
