import numpy as np
import pandas as pd
import shap
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from pycaret.classification import interpret_model
from src.modeling import (
    select_best_model,
    evaluate_model_performance,
    check_feature_correlation,
    plot_model_learning_curve
)
from src.utils.chatgpt_utils import get_chatgpt_feedback
from src.utils.generate_prompt import modeling_prompt

# Ensure all dependencies are ready
#check_required_state(["sampled_df", "target_column"], "Modeling")
st.session_state.advice_step = False
df = st.session_state.sampled_df
target = st.session_state.target_column

if "modeling_done" not in st.session_state:
    st.session_state.modeling_done = False

if "shap_done" not in st.session_state:
    st.session_state.shap_done = False


model_choice = st.radio(
    "Choose modeling approach:",
    ("Train a new model", "Upload a pre-trained model")
)

uploaded_model = None
if model_choice == "Upload a pre-trained model":
    uploaded_file = st.file_uploader("Upload model file (.pkl or .joblib)", type=["pkl", "joblib"])
    if uploaded_file is not None:
        try:
            import pickle, joblib
            if uploaded_file.name.endswith(".pkl"):
                uploaded_model = pickle.load(uploaded_file)
            else:
                uploaded_model = joblib.load(uploaded_file)
            st.success(f"Model loaded: {uploaded_model.__class__.__name__}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")



if st.button("Apply Models"):
    st.session_state.stepModels = True
    st.session_state.modeling_done = False
    st.session_state.shap_done = False

if st.session_state.get("stepModels"):
    X = df.drop(columns=[target])
    y = df[target]
    st.session_state.X = X
    st.session_state.y = y

    with st.spinner("🔍 Checking for highly correlated features..."):
        correlated_features = check_feature_correlation(X, threshold=0.95)
        st.info(f"Detected {len(correlated_features)} highly correlated features (if any).")

    if not st.session_state.get("modeling_done", False):
        if model_choice == "Upload a pre-trained model" and uploaded_model is not None:
            st.session_state.selected_model = uploaded_model
            st.session_state.test_acc = st.number_input("Enter test accuracy (optional):", 0.0, 1.0, value=0.0)
            st.session_state.model_scores = {}
            st.session_state.modeling_done = True
            st.success(f"✅ Using uploaded model: {uploaded_model.__class__.__name__}")

        elif model_choice == "Train a new model":
            with st.spinner("⚙️ Running AutoML to select the best model..."):
                selected_model, model_scores, test_acc = select_best_model(X, y)
                st.session_state.model_scores = model_scores
                st.session_state.test_acc = test_acc
                st.session_state.selected_model = selected_model
                st.success(f"✅ Model selected: {selected_model.__class__.__name__}")
                st.session_state.modeling_done = True

    # Feature importance (tree-based models only)
    st.subheader("🔎 Feature Importance Analysis")
    tree_based_models = ["xgboost", "rf", "et", "dt", "lightgbm"]
    model_name = st.session_state.selected_model.__class__.__name__.lower()
    if any(alias in model_name for alias in tree_based_models):
        with st.spinner("📊 Generating feature importance chart..."):
            interpret_model(st.session_state.selected_model)
    else:
        st.warning("⚠️ Feature importance only available for tree-based models.")


if uploaded_model is not None or st.session_state.modeling_done is not False:
    # Learning Curve
    st.subheader("📈 Learning Curve Analysis")
    with st.spinner("📉 Plotting learning curve..."):
        figModel = plot_model_learning_curve(
            st.session_state.selected_model, st.session_state.X, st.session_state.y,
            title=f"Learning Curve for {st.session_state.selected_model.__class__.__name__}"
        )
        st.session_state.figModel = figModel
        st.pyplot(st.session_state.figModel)

    # Model Scores
    st.subheader("📊 Model Performance")
    st.write(f"✅ Model Parameters:")
    st.write(st.session_state.selected_model.get_params())
    st.write(f"📊 Comparison Scores: {st.session_state.model_scores}")
    st.markdown(f"**🧪 Test Accuracy:** `{st.session_state.test_acc:.3f}`")

    with st.spinner("🔁 Running cross-validation..."):
        cv_scores = cross_val_score(st.session_state.selected_model, st.session_state.X, st.session_state.y, cv=5, scoring='accuracy')
        st.info(f"Mean Cross-Validation Accuracy: `{np.mean(cv_scores):.4f}`")

    # Tree-based feature importances (optional)
    if hasattr(st.session_state.selected_model, "feature_importances_"):
        st.subheader("🔎 Raw Feature Importances")
        importances = pd.Series(st.session_state.selected_model.feature_importances_, index=st.session_state.X.columns).sort_values(ascending=False)
        st.bar_chart(importances)

    # SHAP Summary Plot
    st.subheader("📊 SHAP Summary Plot")
    if not st.session_state.get("shap_done", False):
        with st.spinner("🧠 Calculating SHAP values..."):
            try:
                explainer = shap.Explainer(st.session_state.selected_model, st.session_state.X)
            except Exception:
                explainer = shap.KernelExplainer(st.session_state.selected_model.predict_proba, shap.sample(st.session_state.X, 100))
            shap_values = explainer(st.session_state.X)
            st.session_state.shap_values = shap_values
        st.session_state.shap_done = True

    with st.spinner("📈 Rendering SHAP plot..."):
        values = st.session_state.shap_values.values
        data = st.session_state.shap_values.data
        feature_names = st.session_state.shap_values.feature_names
        base_values = st.session_state.shap_values.base_values

        shape = values.shape
        num_dims = len(shape)

        if num_dims == 3:
            n_samples, n_features, n_classes = shape
            st.text(f"SHAP shape: {shape}, Data shape: {data.shape}")
            class_index = 1 if n_classes == 2 else st.selectbox("Select class to visualize", range(n_classes))
            vals = values[:, :, class_index]
            if vals.shape[1] != data.shape[1]:
                vals = vals[:, :-1]
            st.session_state.expl = shap.Explanation(
                values=vals,
                base_values=base_values[:, class_index] if base_values.ndim == 2 else base_values,
                data=data,
                feature_names=feature_names
            )

            shap_summary = dict(zip(
                st.session_state.expl.feature_names,
                np.mean(np.abs(st.session_state.expl.values), axis=0)
            ))
            st.info(f"SHAP Summary for class {class_index}")
        elif num_dims == 2:
            if values.shape[1] != data.shape[1]:
                values = values[:, :-1]
            st.session_state.expl = shap.Explanation(
                values=values,
                base_values=base_values,
                data=data,
                feature_names=feature_names
            )
            st.session_state.shap_summary = dict(zip(
                st.session_state.expl.feature_names,
                np.mean(np.abs(st.session_state.expl.values), axis=0)
            ))
            st.info("SHAP Summary for binary classification")
        else:
            st.error(f"❌ Unexpected SHAP value shape: {shape}")
            st.session_state.expl = None

        if st.session_state.expl is not None:

            st.session_state.fig, st.session_state.ax = plt.subplots()
            shap.plots.beeswarm(st.session_state.expl, show=False)
            st.pyplot(st.session_state.fig)
            plt.clf()

    # Evaluation
    st.subheader("📌 Classification Evaluation")
    with st.spinner("📐 Calculating precision, recall, ROC and confusion matrix..."):
        st.session_state.precision, recall, cm_fig, roc_fig = evaluate_model_performance(st.session_state.selected_model, st.session_state.X, st.session_state.y)

    st.markdown(f"**🎯 Precision:** `{st.session_state.precision:.3f}`")
    st.markdown(f"**🎯 Recall:** `{recall:.3f}`")
    st.plotly_chart(cm_fig)

    if roc_fig:
        st.pyplot(roc_fig)
    else:
        st.info("⚠️ ROC curve only available for binary classification.")

    # Feature correlations
    st.subheader("🔁 Feature-Target Correlations")
    correlations = df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
    st.dataframe(correlations)

    # Build SHAP summary
    shap_summary = dict(zip(feature_names, np.mean(np.abs(values), axis=0)))

    # Feature correlations
    target_corrs = df.corr()[target].drop(target).sort_values(key=abs, ascending=False).to_dict()

    st.session_state.advice_step = True

    # Generate prompt
    prompt = modeling_prompt(
        model_name=st.session_state.selected_model.__class__.__name__,
        model_params=st.session_state.selected_model.get_params(),
        test_accuracy=st.session_state.test_acc,
        cv_scores=cv_scores.tolist(),
        precision=st.session_state.precision,
        recall=recall,
        shap_summary=shap_summary,
        top_feature_corrs=target_corrs,
        correlated_features=correlated_features,
        n_classes=len(np.unique(y))
    )

    st.session_state.prompt = prompt

if st.session_state.advice_step:
    # Button to get feedback
    if st.button("🧠 Generate Clinical Model Feedback"):
        with st.spinner("Getting AI feedback..."):
            response = get_chatgpt_feedback(st.session_state.prompt)
            st.markdown("### 💬 AI Feedback")
            st.write(response)