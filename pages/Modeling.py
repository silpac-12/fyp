import numpy as np
import pandas as pd
import shap
import streamlit as st
from matplotlib import pyplot as plt, get_cachedir
from sklearn.model_selection import cross_val_score, train_test_split
from pycaret.classification import interpret_model
from sklearn.pipeline import Pipeline

from src.modeling import (
    evaluate_model_performance,
    check_feature_correlation,
    plot_model_learning_curve,
    get_top_models,
    finalize_chosen_model,
    get_final_estimator, finalize_uploaded_model, get_cached_top_models
)
from src.utils.chatgpt_utils import get_chatgpt_feedback
from src.utils.generate_prompt import modeling_prompt

st.title("Modelling Page")

# ---------------------------
# Session state initialization
# ---------------------------
def init_session():
    defaults = {
        "modeling_done": False,
        "shap_done": False,
        "selected_model": None,
        "model_choice": "Train a new model",
        "chosen_model_name": None,
        "stepModels": False,
        "model_scores": {},
        "test_acc": 0.0,
        "precision": None,
        "prompt": None,
        "advice_step": False
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_session()
st.session_state.advice_step = False

# ---------------------------
# Load dataset
# ---------------------------
if "sampled_df" not in st.session_state:
    st.warning("Please upload a dataset in the Home page or complete sampling.")
else:
    df = st.session_state.sampled_df
    target = st.session_state.target_column
    st.session_state.model_choice = st.radio(
        "Choose modeling approach:",
        ("Train a new model", "Upload a pre-trained model")
    )

uploaded_model = None
if st.session_state.model_choice == "Upload a pre-trained model":
    uploaded_file = st.file_uploader("Upload model file (.pkl or .joblib)", type=["pkl", "joblib"])
    if uploaded_file is not None:
        try:
            import pickle, joblib
            uploaded_model = pickle.load(uploaded_file) if uploaded_file.name.endswith(".pkl") else joblib.load(uploaded_file)
            st.success(f"Model loaded: {uploaded_model.__class__.__name__}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

if "sampled_df" in st.session_state:
    if st.button("Apply Models"):
        st.session_state.update({
            "stepModels": True,
            "modeling_done": False,
            "shap_done": False,
            "selected_model": None
        })


if st.session_state.stepModels:
    X = df.drop(columns=[target])
    y = df[target]
    st.session_state.X = X
    st.session_state.y = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    st.session_state.update({"X": X_train, "y": y_train, "X_test": X_test, "y_test": y_test})

    with st.spinner("🔍 Checking for highly correlated features..."):
        correlated_features = check_feature_correlation(X, threshold=0.95)
        st.info(f"Detected {len(correlated_features)} highly correlated features (if any).")

    if not st.session_state.modeling_done:
        if st.session_state.model_choice == "Upload a pre-trained model" and uploaded_model is not None:
            # Extract model if in a pipeline
            if isinstance(uploaded_model, Pipeline):
                final_model = uploaded_model.steps[-1][1]
            else:
                final_model = uploaded_model

            # Set the selected model for Streamlit session
            st.session_state.selected_model = uploaded_model  # keep the full pipeline for prediction

            # Evaluate accuracy and metrics
            selected_model, model_scores, test_acc = finalize_uploaded_model(
                uploaded_model,
                st.session_state.X_test,
                st.session_state.y_test
            )

            st.session_state.model_scores = model_scores
            st.session_state.test_acc = test_acc
            st.session_state.modeling_done = True
            st.success(f"✅ Using uploaded model: {get_final_estimator(st.session_state.selected_model)}")

        elif st.session_state.model_choice == "Train a new model":
            with st.spinner("⚙️ Running AutoML to select the best model..."):
                model_map, comparison_df = get_cached_top_models(X_train, y_train)
                model_names = list(model_map.keys())

            st.subheader("Select Between top 5 models from PyCaret")
            st.write("Models are ranked in order of accuracy")

            # Persistent dropdown
            default_index = model_names.index(st.session_state.chosen_model_name) if st.session_state.chosen_model_name in model_names else 0
            st.session_state.chosen_model_name = st.selectbox("🧠 Choose one of the top models:", model_names, index=default_index)

            if st.button("Final Selection"):
                selected_model = model_map[st.session_state.chosen_model_name]
                selected_model, model_scores, test_acc = finalize_chosen_model(
                    selected_model,
                    comparison_df,
                    st.session_state.chosen_model_name
                )
                st.session_state.update({
                    "selected_model": selected_model,
                    "model_scores": model_scores,
                    "test_acc": test_acc,
                    "modeling_done": True
                })

    # Model info
    if st.session_state.selected_model is not None and st.session_state.modeling_done:

        st.success(f"Finalized model: {get_final_estimator(st.session_state.selected_model)} with accuracy: {st.session_state.test_acc:.3f}")
        #st.success(f"✅ Model selected: {st.session_state.selected_model.__class__.__name__}")

        st.subheader("🔎 Feature Importance Analysis")
        model_name = st.session_state.selected_model.__class__.__name__.lower()
        if any(alias in model_name for alias in ["xgboost", "rf", "et", "dt", "lightgbm"]):
            with st.spinner("📊 Generating feature importance chart..."):
                interpret_model(st.session_state.selected_model)
        else:
            st.warning("⚠️ Feature importance only available for tree-based models.")

# ---------------------------
# Evaluation + SHAP + Prompt
# ---------------------------
if uploaded_model is not None or st.session_state.modeling_done:
    st.subheader("📈 Learning Curve Analysis")
    with st.spinner("📉 Plotting learning curve..."):
        figModel = plot_model_learning_curve(
            st.session_state.selected_model, st.session_state.X, st.session_state.y,
            title=f"Learning Curve for {get_final_estimator(st.session_state.selected_model)}"
        )
        st.pyplot(figModel)

    st.subheader("Model Performance")
    st.write("✅ Model Parameters:")
    st.write(st.session_state.selected_model.get_params())
    st.write("📊 Comparison Scores:", st.session_state.model_scores)
    st.markdown(f"**🧪 Test Accuracy:** `{st.session_state.test_acc:.3f}`")

    with st.spinner("🔁 Running cross-validation..."):
        cv_scores = cross_val_score(st.session_state.selected_model, st.session_state.X, st.session_state.y, cv=5, scoring='accuracy')
        st.write("5 Folds")
        st.success(f"Mean Cross-Validation Accuracy: `{np.mean(cv_scores):.4f}`")

    st.subheader("Classification Evaluation")
    with st.spinner("Calculating precision, recall, ROC and confusion matrix..."):
        st.session_state.precision, recall, cm_fig, roc_fig = evaluate_model_performance(
            st.session_state.selected_model, st.session_state.X_test, st.session_state.y_test
        )

    st.info(f"**🎯 Precision:** `{st.session_state.precision:.3f}`")
    st.info(f"**🎯 Recall:** `{recall:.3f}`")
    st.plotly_chart(cm_fig)
    if roc_fig:
        st.pyplot(roc_fig)
    else:
        st.info("⚠️ ROC curve only available for binary classification.")

    st.subheader("Feature-Target Correlations")
    correlations = df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
    st.dataframe(correlations)

    st.subheader("📊 SHAP Summary Plot")
    if not st.session_state.shap_done:
        with st.spinner("🧠 Calculating SHAP values..."):
            try:
                explainer = shap.Explainer(st.session_state.selected_model, st.session_state.X)
            except Exception:
                safe_sample = st.session_state.X.sample(n=100, random_state=42)
                explainer = shap.KernelExplainer(
                    lambda x: st.session_state.selected_model.predict_proba(pd.DataFrame(x, columns=st.session_state.X.columns)),
                    safe_sample
                )
            shap_values = explainer(st.session_state.X)
            st.session_state.shap_values = shap_values
            st.session_state.shap_done = True

    with st.spinner("📈 Rendering SHAP plot..."):
        shap_values = st.session_state.shap_values
        values = shap_values.values
        data = shap_values.data
        feature_names = shap_values.feature_names
        base_values = shap_values.base_values

        if len(values.shape) == 3:
            class_index = 1 if values.shape[2] == 2 else st.selectbox("Select class to visualize", range(values.shape[2]))
            vals = values[:, :, class_index]
            expl = shap.Explanation(
                values=vals,
                base_values=base_values[:, class_index] if base_values.ndim == 2 else base_values,
                data=data,
                feature_names=feature_names
            )
        else:
            vals = values if values.shape[1] == data.shape[1] else values[:, :-1]
            expl = shap.Explanation(
                values=vals,
                base_values=base_values,
                data=data,
                feature_names=feature_names
            )

        st.session_state.expl = expl
        fig, ax = plt.subplots()
        shap.plots.beeswarm(expl, show=False)
        st.pyplot(fig)
        plt.clf()

    shap_summary = dict(zip(feature_names, np.mean(np.abs(values), axis=0)))
    target_corrs = correlations.to_dict()
    st.session_state.advice_step = True

    prompt = modeling_prompt(
        model_name=get_final_estimator(st.session_state.selected_model),
        model_params=st.session_state.selected_model.get_params(),
        test_accuracy=st.session_state.test_acc,
        cv_scores=cv_scores.tolist(),
        precision=st.session_state.precision,
        recall=recall,
        shap_summary=shap_summary,
        top_feature_corrs=target_corrs,
        correlated_features=correlated_features,
        n_classes=len(np.unique(st.session_state.y))
    )
    st.session_state.prompt = prompt

if st.session_state.advice_step:
    if st.button("🧠 Generate Clinical Model Feedback"):
        with st.spinner("Getting AI feedback..."):
            response = get_chatgpt_feedback(st.session_state.prompt)
            st.markdown("### 💬 AI Feedback")
            st.write(response)
