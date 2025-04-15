import numpy as np
import pandas as pd
import streamlit as st
#from pages.Preprocessing import df_imputed
from src.SMOTE_checker import check_smote_applicability, apply_smote, apply_undersampling
from src.imputation import decode_categorical

from src.utils.utils import initialize_session_state

initialize_session_state({
    "df_final": None,
    "df_imputed": None,
    "target_column": None,
    "stepSampling": False,
    "stepApplySample": False,
    "sampling_method": "No Sampling",
    "sampling_scores": None,
    "clf": None,
    "learning_curve_plot": None,
})

if st.session_state.df_imputed is None:
    st.warning("Please complete pre-processing")

df_final = st.session_state.get("df_final")
df_imputed = st.session_state.get("df_imputed")
target = st.session_state.get("target_column")

if st.session_state.df_final is not None:

    st.header("Class Balancing")

    # Dropdown to select the target variable
    #target_column = st.selectbox("Select Target Variable:", st.session_state.df.columns)

    if "target_column" in st.session_state and st.session_state.target_column:
        default_target = st.session_state.target_column
    else:
        default_target = st.session_state.df.columns[0]

    target_column = st.selectbox("Select Target Variable:", st.session_state.df.columns,
                                 index=st.session_state.df.columns.get_loc(default_target))
    st.session_state.target_column = target_column


    # Ensure the selected target variable is displayed
    st.write(f"### Selected Target Variable: `{target_column}`")
    st.write("Unique Values in Target Column:", st.session_state.df[target_column].unique())

    X = st.session_state.df_imputed.drop(columns=target_column)
    y = pd.Series(st.session_state.df_imputed[target_column], name=target_column)

    if st.button("Target Class Distribution"):
        fig, result = check_smote_applicability(st.session_state.df_final, st.session_state.target_column)
        st.session_state.smote_fig = fig
        st.session_state.smote_result = result

    # Display only if previously run
    if st.session_state.get("smote_fig"):
        st.pyplot(st.session_state.smote_fig)

    if st.session_state.get("smote_result"):
        st.json(st.session_state.smote_result)
        st.success(st.session_state.smote_result["recommendation"])

    st.session_state.stepSampling = True

if st.session_state.stepSampling:

    # ✅ Ensure sampling scores persist
    if st.button("Evaluate Sampling"):
        from src.SMOTE_checker import evaluate_sampling_methods
        st.session_state.df_final = st.session_state.df_imputed
        X = st.session_state.df_final.drop(columns=[st.session_state.target_column])
        y = st.session_state.df_final[st.session_state.target_column]

        # Convert `X` to purely numeric
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.select_dtypes(include=[np.number])  # Remove non-numeric data

        # Convert `y` to integer
        y = y.astype(int)
        st.session_state.sampling_scores, trained_clf = evaluate_sampling_methods(X, y, st.session_state.sampling_method)
        st.session_state.clf = trained_clf



    # Fix for Sampling Selection Reset
    st.subheader("📊 Sampling Strategy Evaluation")

    # Display stored scores even after script reruns
    if st.session_state.sampling_scores:
        st.json(st.session_state.sampling_scores)
        st.session_state.stepApplySample = True

    # Ensure learning curve persists
    st.subheader("📈 Learning Curve")
    if st.button("Plot Learning Curve"):
        from src.SMOTE_checker import plot_learning_curve
        X = st.session_state.df_final.drop(columns=[st.session_state.target_column])
        y = st.session_state.df_final[st.session_state.target_column]
        st.session_state.learning_curve_plot = plot_learning_curve(st.session_state.clf, X, y)

    # ✅ Show the learning curve if available
    if st.session_state.learning_curve_plot:
        st.pyplot(st.session_state.learning_curve_plot)



if st.session_state.stepApplySample:

    prev_method = st.session_state.get("sampling_method", "No Sampling")

    selected_method = st.selectbox(
        "Choose a Sampling Method",
        ["No Sampling", "SMOTE", "Undersampling"],
        index=["No Sampling", "SMOTE", "Undersampling"].index(prev_method)
    )


    #st.write("Selected Method: ", st.session_state.sampling_method)
    if selected_method != prev_method or "sampled_df" not in st.session_state:
        X = st.session_state.df_final.drop(columns=[st.session_state.target_column])
        y = st.session_state.df_final[st.session_state.target_column]

        if selected_method == "SMOTE":
            sampled_df = apply_smote(X, y)
        elif selected_method == "Undersampling":
            sampled_df = apply_undersampling(X, y)
        else:
            sampled_df = st.session_state.df_final

        st.session_state.sampled_df = sampled_df
        st.session_state.sampling_method = selected_method

    # Store the selected dataset in session state
    #st.session_state.sampled_df = sampled_df

        sampled_decoded_df = decode_categorical(sampled_df, st.session_state.mappings)
        #Display Selected Dataset
        st.write(f"✅ **Dataset Preview - {selected_method}**")
        st.dataframe(sampled_decoded_df)
        fig1, result1 = check_smote_applicability(sampled_df, st.session_state.target_column)
        if fig1:
            st.pyplot(fig1)
        if result1:
            st.json(result1)

        # ✅ Provide Download Option

        csv = sampled_decoded_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Processed Dataset", csv, f"{selected_method}_data.csv", "text/csv")

        # ✅ Update session state immediately and rerun
        if selected_method != st.session_state.sampling_method:
            st.session_state.sampling_method = selected_method
            #st.rerun()

