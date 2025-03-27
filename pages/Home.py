import pandas as pd
import streamlit as st
from src.data_loader import load_dataset, check_missing_values
from src.feature_analysis import show_summary, plot_correlation_heatmap
from src.imputation import encode_categorical, apply_imputation, decode_categorical
from src.utils.chatgpt_utils import get_chatgpt_feedback
from src.utils.generate_prompt import eda_prompt, imputation_prompt
from src.utils.utils import initialize_session_state

st.title("Cancer Prediction - Data Processing & Imputation")

initialize_session_state({
    "df": None,
    "dropped_columns_df": None,
    "mappings": None,
    "df_final": None,
    "sampling_method": "No Sampling",
    "sampling_scores": None,
    "learning_curve_plot": None,
    "stepImputation": False,
    "stepModels": False,
    "stepSampling": False,
    "stepApplySample": False,
    "stepTarget": False,
})

# ✅ File uploader - Ensure df is not reset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None and st.session_state.df is None:
    st.session_state.df = load_dataset(uploaded_file)

if st.session_state.df is not None:
    df = st.session_state.df
    st.write("🔹 **Original Dataset Loaded:**")
    st.write(df.head())
    missing_summary = check_missing_values(df)
    st.write(check_missing_values(df))

    # Column selection and removal
    st.subheader("🔻 Remove Columns (e.g., IDs, Dates)")
    cols_to_drop = st.multiselect("Select columns to remove", df.columns)
    st.session_state.cols_to_drop = cols_to_drop

    if st.button("Remove Selected Columns"):
        st.session_state.dropped_columns_df = df[cols_to_drop].copy()
        df = df.drop(columns=cols_to_drop)
        st.session_state.df = df
        st.success(f"✅ Removed columns: {cols_to_drop}")
        st.write("❌ **Dataset after column removal:**")
        st.write(df.head())
        st.session_state.stepTarget = True

    st.write(show_summary(df))

    # Encoding Step
    st.subheader("🔄 Encode Categorical Variables")
    df_encoded, mappings = encode_categorical(df)
    st.session_state.df_encoded = df_encoded
    st.session_state.mappings = mappings
    st.write("✅ **After Encoding:**")
    st.write(df_encoded.head())
    corr_mat = plot_correlation_heatmap(df_encoded)
    st.write(corr_mat)
    corr_mat2 = df_encoded.corr()



