import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.SMOTE_checker import check_smote_applicability
from src.data_loader import load_dataset, check_missing_values
from src.feature_analysis import plot_histograms, show_summary, compare_feature_means, \
    compare_missing_values, compare_correlation_matrices, compare_feature_stds, plot_correlation_heatmap
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
    st.write(plot_correlation_heatmap(df))
    st.write(plot_histograms(df))
    st.write(show_summary(df))

    # Encoding Step
    st.subheader("🔄 Encode Categorical Variables")
    df_encoded, mappings = encode_categorical(st.session_state.df)
    st.session_state.df_encoded = df_encoded
    st.session_state.mappings = mappings
    st.write("✅ **After Encoding:**")
    #df_encoded = df_encoded.drop(columns=cols_to_drop)
    st.write(df_encoded.head())

    # Imputation Step
    st.subheader("🩺 Impute Missing Values")
    imputation_method = st.selectbox("Choose an imputation method", ["mean", "zero", "mice"])
    df_before = df_encoded
    if st.button("Apply Imputation"):
        # Apply imputation
        df_imputed = apply_imputation(df_encoded, imputation_method, mappings)
        df_after = df_imputed
        #st.write("✅ **After Imputation:**")
        #st.write(df_imputed.head())

        # Decode categorical columns
        df_decoded = decode_categorical(df_imputed, mappings)
        #st.write("🔄 **After Decoding:**")
        #st.write(df_decoded.head())

        # Optionally reattach dropped columns
        if st.checkbox("🔗 Reattach Dropped Columns"):
            df_final = pd.concat([st.session_state.dropped_columns_df.reset_index(drop=True),
                                  df_decoded.reset_index(drop=True)], axis=1)
        else:
            df_final = df_decoded  # Keep columns removed if not selected

        st.success("✅ **Final Dataset Ready!**")
        df_final = df_final.drop(columns=cols_to_drop)
        df = df.drop(columns=cols_to_drop)
        st.write(df_final.head())

        # Download option
        csv = df_final.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Processed Dataset", csv, "imputed_data.csv", "text/csv")

        # Assuming df_before and df_after are defined in your app
        df_before = df_before.drop(columns=cols_to_drop)
        df_after = df_after.drop(columns=cols_to_drop)
        # 🔢 Feature Means Comparison
        st.subheader("Feature Means Comparison")
        means_df = compare_feature_means(df_before, df_after)
        st.dataframe(means_df)

        # 📊 Feature Standard Deviations Comparison
        st.subheader("Feature Standard Deviations Comparison")
        stds_df = compare_feature_stds(df_before, df_after)
        st.dataframe(stds_df)

        # 🕳️ Missing Values Comparison
        st.subheader("Missing Values Comparison")
        missing_df = compare_missing_values(df_before, df_after)
        st.dataframe(missing_df)

        # 🧭 Correlation Matrices Comparison (Heatmaps Only)
        st.subheader("Correlation Matrices Comparison")
        st.pyplot(compare_correlation_matrices(df_before, df_after))

        st.header("Class Balancing")
        fig, result = check_smote_applicability(df_after, "case_csPCa")
        # Display the plot in Streamlit
        if fig:
            st.pyplot(fig)  # Correct way to display Matplotlib figures in Streamlit

        # Display the result as JSON or formatted text
        if result:
            st.json(result)  # Nicely formatted JSON display
            st.success(result["recommendation"])  # Display the recommendation message
