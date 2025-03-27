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

df = st.session_state.sampled_df
target = st.session_state.target_column

if st.button("Apply Models"):
    st.session_state.stepModels = True

if st.session_state.get("stepModels"):
    X = df.drop(columns=[target])
    y = df[target]

    with st.spinner("🔍 Checking for highly correlated features..."):
        correlated_features = check_feature_correlation(X, threshold=0.95)
        st.info(f"Detected {len(correlated_features)} highly correlated features (if any).")

    with st.spinner("⚙️ Running AutoML to select the best model..."):
        selected_model, model_scores, test_acc = select_best_model(X, y)
        st.session_state.selected_model = selected_model
        st.success(f"✅ Model selected: {selected_model.__class__.__name__}")

    # Feature importance (tree-based models only)
    st.subheader("🔎 Feature Importance Analysis")
    tree_based_models = ["xgboost", "rf", "et", "dt", "lightgbm"]
    model_name = selected_model.__class__.__name__.lower()
    if any(alias in model_name for alias in tree_based_models):
        with st.spinner("📊 Generating feature importance chart..."):
            interpret_model(selected_model)
    else:
        st.warning("⚠️ Feature importance only available for tree-based models.")

    # Learning Curve
    st.subheader("📈 Learning Curve Analysis")
    with st.spinner("📉 Plotting learning curve..."):
        figModel = plot_model_learning_curve(
            selected_model, X, y,
            title=f"Learning Curve for {selected_model.__class__.__name__}"
        )
        st.pyplot(figModel)

    # Model Scores
    st.subheader("📊 Model Performance")
    st.write(f"✅ Model Parameters:")
    st.write(selected_model.get_params())
    st.write(f"📊 Comparison Scores: {model_scores}")
    st.markdown(f"**🧪 Test Accuracy:** `{test_acc:.3f}`")

    with st.spinner("🔁 Running cross-validation..."):
        cv_scores = cross_val_score(selected_model, X, y, cv=5, scoring='accuracy')
        st.info(f"Mean Cross-Validation Accuracy: `{np.mean(cv_scores):.4f}`")

    # Tree-based feature importances (optional)
    if hasattr(selected_model, "feature_importances_"):
        st.subheader("🔎 Raw Feature Importances")
        importances = pd.Series(selected_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(importances)

    # SHAP Summary Plot
    st.subheader("📊 SHAP Summary Plot")
    with st.spinner("🧠 Calculating SHAP values..."):
        try:
            explainer = shap.Explainer(selected_model, X)
        except Exception:
            explainer = shap.KernelExplainer(selected_model.predict_proba, shap.sample(X, 100))
        shap_values = explainer(X)

    with st.spinner("📈 Rendering SHAP plot..."):
        values = shap_values.values
        data = shap_values.data
        feature_names = shap_values.feature_names
        base_values = shap_values.base_values

        shape = values.shape
        num_dims = len(shape)

        if num_dims == 3:
            n_samples, n_features, n_classes = shape
            st.text(f"SHAP shape: {shape}, Data shape: {data.shape}")
            class_index = 1 if n_classes == 2 else st.selectbox("Select class to visualize", range(n_classes))
            vals = values[:, :, class_index]
            if vals.shape[1] != data.shape[1]:
                vals = vals[:, :-1]
            expl = shap.Explanation(
                values=vals,
                base_values=base_values[:, class_index] if base_values.ndim == 2 else base_values,
                data=data,
                feature_names=feature_names
            )
            shap_summary = dict(zip(
                expl.feature_names,
                np.mean(np.abs(expl.values), axis=0)
            ))
            st.info(f"SHAP Summary for class {class_index}")
        elif num_dims == 2:
            if values.shape[1] != data.shape[1]:
                values = values[:, :-1]
            expl = shap.Explanation(
                values=values,
                base_values=base_values,
                data=data,
                feature_names=feature_names
            )
            shap_summary = dict(zip(
                expl.feature_names,
                np.mean(np.abs(expl.values), axis=0)
            ))
            st.info("SHAP Summary for binary classification")
        else:
            st.error(f"❌ Unexpected SHAP value shape: {shape}")
            expl = None

        if expl is not None:
            fig, ax = plt.subplots()
            shap.plots.beeswarm(expl, show=False)
            st.pyplot(fig)
            plt.clf()

    # Evaluation
    st.subheader("📌 Classification Evaluation")
    with st.spinner("📐 Calculating precision, recall, ROC and confusion matrix..."):
        precision, recall, cm_fig, roc_fig = evaluate_model_performance(selected_model, X, y)

    st.markdown(f"**🎯 Precision:** `{precision:.3f}`")
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

    # Generate prompt
    prompt = modeling_prompt(
        model_name=selected_model.__class__.__name__,
        model_params=selected_model.get_params(),
        test_accuracy=test_acc,
        cv_scores=cv_scores.tolist(),
        precision=precision,
        recall=recall,
        shap_summary=shap_summary,
        top_feature_corrs=target_corrs,
        correlated_features=correlated_features,
        n_classes=len(np.unique(y))
    )

    # Button to get feedback
    if st.button("🧠 Generate Clinical Model Feedback"):
        with st.spinner("Getting AI feedback..."):
            response = get_chatgpt_feedback(prompt)
            st.markdown("### 💬 AI Feedback")
            st.write(response)
