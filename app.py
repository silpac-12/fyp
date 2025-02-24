import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_loader import load_dataset, check_missing_values
from src.imputation import encode_categorical, decode_categorical, apply_imputation

st.title("Cancer Prediction - Data Processing & Imputation")

# Session state initialization
if "df" not in st.session_state:
    st.session_state.df = None
if "dropped_columns_df" not in st.session_state:
    st.session_state.dropped_columns_df = None
if "mappings" not in st.session_state:
    st.session_state.mappings = None

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = load_dataset(uploaded_file)
    st.session_state.df = df

    st.write("🔹 **Original Dataset Loaded:**")
    st.write(df.head())
    st.write(check_missing_values(df))

    # Column selection and removal
    st.subheader("🔻 Remove Columns (e.g., IDs, Dates)")
    cols_to_drop = st.multiselect("Select columns to remove", df.columns)

    if st.button("Remove Selected Columns"):
        # Save dropped columns separately
        st.session_state.dropped_columns_df = df[cols_to_drop].copy()
        # Remove selected columns from df
        df = df.drop(columns=cols_to_drop)
        st.session_state.df = df
        st.success(f"✅ Removed columns: {cols_to_drop}")
        st.write("❌ **Dataset after column removal:**")
        st.write(df.head())

    #Feature Analysis

    # Encoding Step
    st.subheader("🔄 Encode Categorical Variables")
    df_encoded, mappings = encode_categorical(st.session_state.df)
    st.session_state.df_encoded = df_encoded
    st.session_state.mappings = mappings
    st.write("✅ **After Encoding:**")
    st.write(df_encoded.head())

    # Imputation Step
    st.subheader("🩺 Impute Missing Values")
    imputation_method = st.selectbox("Choose an imputation method", ["mean", "zero", "mice"])

    if st.button("Apply Imputation"):
        # Apply imputation
        df_imputed = apply_imputation(df_encoded, imputation_method, mappings)
        st.write("✅ **After Imputation:**")
        st.write(df_imputed.head())

        # Decode categorical columns
        df_decoded = decode_categorical(df_imputed, mappings)
        st.write("🔄 **After Decoding:**")
        st.write(df_decoded.head())

        # Optionally reattach dropped columns
        if st.checkbox("🔗 Reattach Dropped Columns"):
            df_final = pd.concat([st.session_state.dropped_columns_df.reset_index(drop=True),
                                  df_decoded.reset_index(drop=True)], axis=1)
        else:
            df_final = df_decoded  # Keep columns removed if not selected

        st.success("✅ **Final Dataset Ready!**")
        df_final = df_final.drop(columns=cols_to_drop)
        st.write(df_final.head())

        # Download option
        csv = df_final.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Processed Dataset", csv, "imputed_data.csv", "text/csv")
