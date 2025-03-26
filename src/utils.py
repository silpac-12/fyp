from turtle import st
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
import numpy as np


def extract_inner_model(model):
    from pycaret.internal.pipeline import Pipeline
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model

def plot_shap_summary(model, X):
    """
    Returns a SHAP summary beeswarm plot figure and a label.
    Works with both binary and multiclass classifiers.
    """

    # Step 1: Explainer
    try:
        explainer = shap.Explainer(model, X)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))

    # Step 2: Get shap_values
    shap_values = explainer(X)

    # Step 3: Normalize shapes
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    data = shap_values.data
    feature_names = shap_values.feature_names

    st.subheader("📊 SHAP Summary Plot")

    # Determine if this is multiclass (3D shap_values)
    is_multiclass = len(shap_values.values.shape) == 3

    if is_multiclass:
        st.info("Detected multiclass classification")

        num_classes = shap_values.values.shape[1]
        class_index = st.selectbox("Select Class to Visualize", range(num_classes))

        # Slice SHAP values for selected class
        shap_vals_for_class = shap_values[..., class_index]

        # 🛠 Optional: fix offset shape mismatch
        if shap_vals_for_class.values.shape[1] != shap_vals_for_class.data.shape[1]:
            shap_vals_for_class.values = shap_vals_for_class.values[:, :-1]

        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_vals_for_class, show=False)
        st.pyplot(fig)
        plt.clf()

    else:
        st.info("Detected binary classification")

        # 🛠 Optional: fix offset shape mismatch
        if shap_values.values.shape[1] != shap_values.data.shape[1]:
            shap_values.values = shap_values.values[:, :-1]

        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)
        plt.clf()