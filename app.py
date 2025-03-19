import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from streamlit import session_state
from pycaret.classification import setup, compare_models, tune_model, interpret_model

from src.SMOTE_checker import check_smote_applicability, evaluate_sampling_methods, apply_smote, apply_undersampling
from src.data_loader import load_dataset, check_missing_values
from src.feature_analysis import plot_histograms, show_summary, compare_feature_means, \
    compare_missing_values, compare_correlation_matrices, compare_feature_stds, plot_correlation_heatmap
from src.imputation import encode_categorical, decode_categorical, apply_imputation
from src.modeling import select_best_model, plot_model_learning_curve, \
    check_feature_correlation

st.title("Cancer Prediction - Data Processing & Imputation")

# ✅ Ensure session state variables are initialized
if "df" not in st.session_state:
    st.session_state.df = None
if "dropped_columns_df" not in st.session_state:
    st.session_state.dropped_columns_df = None
if "mappings" not in st.session_state:
    st.session_state.mappings = None
if "df_final" not in st.session_state:
    st.session_state.df_final = None
if "sampling_method" not in st.session_state:
    st.session_state.sampling_method = "No Sampling"
if "sampling_scores" not in st.session_state:
    st.session_state.sampling_scores = None
if "learning_curve_plot" not in st.session_state:
    st.session_state.learning_curve_plot = None
if "stepImputation" not in st.session_state:
    st.session_state.stepImputation = False
if "stepModels" not in st.session_state:
    st.session_state.stepModels = False
if "stepSampling" not in st.session_state:
    st.session_state.stepSampling = False
if "stepApplySample" not in st.session_state:
    st.session_state.stepApplySample = False

# ✅ File uploader - Ensure df is not reset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None and st.session_state.df is None:
    st.session_state.df = load_dataset(uploaded_file)

if st.session_state.df is not None:
    df = st.session_state.df
    st.write("🔹 **Original Dataset Loaded:**")
    st.write(df.head())
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


    # Feature Analysis
    #st.pyplot(plot_histograms(df))
    st.write(plot_histograms(df))
    st.write(show_summary(df))

    # Encoding Step
    st.subheader("🔄 Encode Categorical Variables")
    df_encoded, mappings = encode_categorical(df)
    st.session_state.df_encoded = df_encoded
    st.session_state.mappings = mappings
    st.write("✅ **After Encoding:**")
    st.write(df_encoded.head())
    st.write(plot_correlation_heatmap(df_encoded))

    # Imputation Step
    st.subheader("🩺 Impute Missing Values")
    imputation_method = st.selectbox("Choose an imputation method", ["mean", "zero", "mice"])

    if st.button("Apply Imputation"):
        df_imputed = apply_imputation(df_encoded, imputation_method, mappings)
        st.session_state.df_imputed = df_imputed.drop(columns=cols_to_drop)
        df_final = decode_categorical(df_imputed, mappings)

        if st.checkbox("🔗 Reattach Dropped Columns"):
            df_final = pd.concat([st.session_state.dropped_columns_df.reset_index(drop=True),
                                  df_final.reset_index(drop=True)], axis=1)

        st.success("✅ **Final Dataset Ready!**")
        df_final = df_final.drop(columns=cols_to_drop)
        st.session_state.df_final = df_final
        st.session_state.stepImputation = True
        st.rerun()


if st.session_state.stepImputation:

    st.write(st.session_state.df_final.head())

    # Download option
    csv = st.session_state.df_final.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Processed Dataset", csv, "imputed_data.csv", "text/csv")

    # Feature Comparisons
    df_before = st.session_state.df_encoded.drop(columns=st.session_state.cols_to_drop)
    df_after = st.session_state.df_imputed.drop(columns=st.session_state.cols_to_drop)
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



# ✅ Ensure `df_final` is available before proceeding
if st.session_state.df_final is not None:

    # Dropdown to select the target variable
    target_column = st.selectbox("Select Target Variable:", st.session_state.df_final.columns)
    st.session_state.target_column = target_column
    # Ensure the selected target variable is displayed
    st.write(f"### Selected Target Variable: `{target_column}`")
    st.write("Unique Values in Target Column:", st.session_state.df_final[target_column].unique())

    st.header("Class Balancing")

    # Call SMOTE analysis function
    fig, result = check_smote_applicability(st.session_state.df_final, st.session_state.target_column)
    if fig:
        st.pyplot(fig)
    if result:
        st.json(result)
        st.success(result["recommendation"])

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
        st.session_state.sampling_scores, trained_clf = evaluate_sampling_methods(X, y,                                                                         st.session_state.sampling_method)
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

    selected_method = st.selectbox(
            "Choose a Sampling Method",
            ["No Sampling", "SMOTE", "Undersampling"],
            index=["No Sampling", "SMOTE", "Undersampling"].index(st.session_state.sampling_method)
    )

    X = st.session_state.df_final.drop(columns=[st.session_state.target_column])
    y = st.session_state.df_final[st.session_state.target_column]

    # Convert `X` to purely numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.select_dtypes(include=[np.number])  # Remove non-numeric data

    # Convert `y` to integer
    #y = y.astype(int)
    sampled_df = st.session_state.df_final
    # ✅ Apply Selected Sampling Method
    if selected_method == "No Sampling":
        sampled_df = st.session_state.df_final
    elif selected_method == "SMOTE":
        sampled_df = apply_smote(X, y)
    elif selected_method == "Undersampling":
        sampled_df = apply_undersampling(X, y)

    # Store the selected dataset in session state
    st.session_state.sampled_df = sampled_df

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
    if st.button("Apply Models"):
        st.session_state.stepModels = True


if st.session_state.stepModels:
    # Extract features (X) and target variable (y)
    X = st.session_state.sampled_df.drop(columns=[st.session_state.target_column])
    y = st.session_state.sampled_df[st.session_state.target_column]

    st.write("X: ", X)
    st.write("Y: ", y)

    # Debug: Print correlation before removing anything
    correlated_features = check_feature_correlation(X, threshold=0.95)
    print(correlated_features)

    # ✅ Run PyCaret’s AutoML to select the best model automatically
    selected_model, model_scores, test_acc = select_best_model(X, y)

    st.write("✅ **Selected Model:**", selected_model.__class__.__name__)

    # 🔹 Feature Importance Analysis
    st.subheader("🔎 Feature Importance Analysis")
    # Check if the selected model is tree-based before interpreting
    tree_based_models = ["xgboost", "rf", "et", "dt", "lightgbm"]

    if selected_model.__class__.__name__.lower() in tree_based_models:
        interpret_model(selected_model)
    else:
        st.warning(
            "Feature importance is only available for tree-based models (XGBoost, Random Forest, Extra Trees, Decision Tree, LightGBM). Skipping interpretation.")
    #interpret_model(selected_model)  # Displays feature importance using PyCaret

    # 🚀 Evaluate the model using cross-validation
    st.subheader("📊 Model Performance")
    st.write(f"✅ Model Scores: {selected_model.get_params()}")  # Display model parameters

    # 🔍 Learning Curve to detect Overfitting
    st.subheader("📈 Learning Curve Analysis")
    figModel = plot_model_learning_curve(selected_model, X, y,
                                         title=f"Learning Curve for {selected_model.__class__.__name__}")
    st.pyplot(figModel)

    st.write(f"✅ Selected Model: {selected_model}")  # Display the model name
    st.write(f"📊 Model Performance: {model_scores}")  # Show model comparison table
    st.write(f"📈 Test Accuracy: {test_acc:.3f}")  # Display test accuracy

    # Run a simple cross-validation test
    cv_scores = cross_val_score(selected_model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")

    # If the selected model supports feature importance, display it
    if hasattr(selected_model, "feature_importances_"):
        st.subheader("🔎 Feature Importance Analysis")
        feature_importances = pd.Series(selected_model.feature_importances_, index=X.columns).sort_values(
            ascending=False)
        st.bar_chart(feature_importances)

    # ✅ Save the trained model (Optional: if you want to use it later)
    from pycaret.classification import save_model

    save_model(selected_model, "best_model")
    st.success("✅ Model saved as `best_model.pkl`")

    for col in X.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=y, y=X[col], ax=ax)
        ax.set_title(f"Feature: {col} vs Target")
        st.pyplot(fig)