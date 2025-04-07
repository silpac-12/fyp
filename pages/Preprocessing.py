import pandas as pd
import streamlit as st
from src.imputation import encode_categorical, apply_imputation, decode_categorical
from src.feature_analysis import plot_correlation_heatmap, compare_feature_means, compare_feature_stds, \
    compare_missing_values, show_summary, compare_correlation_matrices, detect_statistical_shifts, \
    detect_outlier_changes
from src.utils.chatgpt_utils import get_chatgpt_feedback
from src.utils.generate_prompt import imputation_prompt

# Imputation Step
st.subheader("🩺 Impute Missing Values")
imputation_method = st.selectbox("Choose an imputation method", ["mean", "zero", "mice"])


if st.button("Apply Imputation"):
    st.session_state.imputation_method = imputation_method
    df_imputed = apply_imputation(st.session_state.df_encoded, imputation_method, st.session_state.mappings)
    st.session_state.df_imputed = df_imputed
    df_final = decode_categorical(df_imputed, st.session_state.mappings)
    st.write(df_final.head())

    if st.checkbox("🔗 Reattach Dropped Columns"):
        df_final = pd.concat([st.session_state.dropped_columns_df.reset_index(drop=True),
        df_final.reset_index(drop=True)], axis=1)

    st.success("✅ **Final Dataset Ready!**")
    #df_final = df_final.drop(columns=st.session_state.cols_to_drop)
    st.session_state.df_final = df_final
    st.session_state.stepImputation = True

if st.session_state.stepImputation:

    # Download option
    csv = st.session_state.df_final.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Processed Dataset", csv, "imputed_data.csv", "text/csv")

    # Feature Comparisons
    df_before = st.session_state.df_encoded
    df_after = st.session_state.df_imputed
    st.subheader("Feature Means Comparison")
    st.dataframe(compare_feature_means(df_before, df_after))
    st.subheader("Feature Standard Deviations Comparison")
    st.dataframe(compare_feature_stds(df_before, df_after))
    st.subheader("Missing Values Comparison")
    st.dataframe(compare_missing_values(df_before, df_after))
    st.subheader("Correlation Matrices Comparison")
    figCorrMat = compare_correlation_matrices(df_before, df_after)
    st.session_state.figCorrMat = figCorrMat
    st.pyplot(st.session_state.figCorrMat)

    # Detect shifts in features after imputation
    stat_shifts = detect_statistical_shifts(df_before, df_after)
    #target_corr_shifts = detect_target_correlation_shifts(df_before, df_after, y)
    outlier_changes = detect_outlier_changes(df_before, df_after)

    st.write("⚠️ **Outlier Changes After Imputation**")
    st.dataframe(outlier_changes)

    prompt = imputation_prompt(
        st.session_state.imputation_method,
        df_before,
        df_after,
        st.session_state.cols_to_drop,
        df_before.corr(),
        df_after.corr()
    )

    if st.button("🧠 Generate Clinical Feedback"):
        with st.spinner("Getting AI feedback..."):
            response = get_chatgpt_feedback(prompt)
            st.markdown("### 💬 AI Feedback")
            st.write(response)

