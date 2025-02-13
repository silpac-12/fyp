import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_loader import load_dataset, preprocess_data, check_missing_values
from src.feature_analysis import show_summary, plot_histograms, plot_correlation_heatmap
from src.imputation import encode_categorical, decode_categorical, apply_imputation

st.title("Cancer Prediction - Data Processing & Imputation")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
mappings = None  # Ensure mappings persist across function calls
encoded = False  # Track encoding status

if uploaded_file is not None:
    df = load_dataset(uploaded_file)

    st.subheader("Dataset Preview (Before Processing)")
    st.write(df.head())

    # Debugging: Show column types before processing
    st.subheader("Column Data Types (Before Processing)")
    st.write(df.dtypes)

    # Column selection for manual removal **before** feature analysis
    st.subheader("Select Columns to Remove (e.g., Dates, IDs)")
    cols_to_drop = st.multiselect("Choose columns to exclude from processing", df.columns)

    # Button to confirm column removal
    if st.button("Remove Selected Columns"):
        df = df.drop(columns=cols_to_drop, errors="ignore")  # Drop selected columns
        st.success(f"Removed columns: {cols_to_drop}")
        st.write(df.head())

        # Show missing values after column removal
        st.subheader("Missing Values")
        st.write(check_missing_values(df))

        # **New Section: Visualizing the Dataset Before Imputation (Grid Layout)**
        st.subheader("Feature Analysis Before Imputation")

        # Grid Layout for Graphs Before Imputation
        col1, col2 = st.columns(2)  # Two columns for better structure

        # Correlation Heatmap
        with col1:
            st.subheader("Correlation Heatmap Before Imputation")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

        # Missing Value Heatmap
        with col2:
            st.subheader("Missing Data Heatmap Before Imputation")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
            st.pyplot(fig)

        # Additional Graphs
        col3, col4 = st.columns(2)

        # Boxplots
        with col3:
            st.subheader("Boxplots of Numerical Features")
            for col in df.select_dtypes(include=["number"]).columns[:5]:  # Limit to 5 for readability
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(y=df[col], ax=ax)
                st.pyplot(fig)

        # Histograms
        with col4:
            st.subheader("Feature Distributions Before Imputation")
            for col in df.select_dtypes(include=["number"]).columns[:5]:  # Limit to 5 for readability
                fig, ax = plt.subplots()
                sns.histplot(df[col], bins=30, kde=True, ax=ax)
                st.pyplot(fig)

    # Apply encoding
    if st.button("Apply Encoding (No Imputation)"):
        encoded = True
        df_encoded, mappings = encode_categorical(df)  # Store mappings for later decoding
        st.success("Categorical encoding applied.")
        st.subheader("Processed Dataset (After Encoding)")
        st.write(df_encoded.head())

        # Provide download button for processed dataset
        csv = df_encoded.to_csv(index=False).encode("utf-8")
        st.download_button("Download Processed Dataset", csv, "encoded_data.csv", "text/csv")

    # Apply imputation
    st.subheader("Impute Missing Values")
    imputation_method = st.selectbox("Choose an imputation method", ["mean", "zero", "mice"])

    if st.button("Apply Imputation"):
        df_imputed = apply_imputation(df, imputation_method, mappings)  # Pass mappings for decoding
        st.success(f"Imputation applied using {imputation_method}.")
        st.subheader("Dataset After Imputation & Before Decoding")
        st.write(df_imputed.head())

        # Ensure categorical decoding occurs correctly
        if encoded and mappings is not None:
            df_imputed = decode_categorical(df_imputed, mappings)
            st.success("Categorical values restored after imputation.")
            st.subheader("Dataset After Imputation & Decoding")
            st.write(df_imputed.head())

        # **New Section: Visualizing the Imputed Dataset in a Grid Layout**
        st.subheader("Feature Analysis After Imputation")

        # Grid Layout for Graphs After Imputation
        col5, col6 = st.columns(2)

        # Correlation Heatmap After Imputation
        with col5:
            st.subheader("Correlation Heatmap After Imputation")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_imputed.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

        # Missing Value Heatmap After Imputation
        with col6:
            st.subheader("Missing Data Heatmap After Imputation")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_imputed.isnull(), cbar=False, cmap="viridis", ax=ax)
            st.pyplot(fig)

        # Additional Graphs
        col7, col8 = st.columns(2)

        # Boxplots After Imputation
        with col7:
            st.subheader("Boxplots of Numerical Features After Imputation")
            for col in df_imputed.select_dtypes(include=["number"]).columns[:5]:  # Limit to 5 for readability
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(y=df_imputed[col], ax=ax)
                st.pyplot(fig)

        # Histograms After Imputation
        with col8:
            st.subheader("Feature Distributions After Imputation")
            for col in df_imputed.select_dtypes(include=["number"]).columns[:5]:  # Limit to 5 for readability
                fig, ax = plt.subplots()
                sns.histplot(df_imputed[col], bins=30, kde=True, ax=ax)
                st.pyplot(fig)

        # Provide download button for processed dataset
        csv = df_imputed.to_csv(index=False).encode("utf-8")
        st.download_button("Download Processed Dataset", csv, "imputed_data.csv", "text/csv")
