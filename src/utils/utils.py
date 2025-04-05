import streamlit as st
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

def initialize_session_state(defaults: dict):
    """
    Initialize session state variables with default values.
    """
    import streamlit as st
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_shap_summary(model, X, st_container):
    """
    Generates SHAP summary plot and returns the matplotlib figure and class label.
    :param model: Trained model (PyCaret or raw model)
    :param X: DataFrame of input features
    :param st_container: streamlit object (usually `st`)
    :return: Tuple (fig, description_text)
    """
    # Step 1: Create explainer
    try:
        explainer = shap.Explainer(model, X)
    except Exception:
        st_container.warning("Defaulting to KernelExplainer — SHAP may be slow.")
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))

    # Step 2: Get SHAP values
    shap_values = explainer(X)

    # Step 3: Extract and normalize
    values = shap_values.values
    data = shap_values.data
    feature_names = shap_values.feature_names
    base_values = shap_values.base_values

    shape = values.shape
    num_dims = len(shape)

    # Step 4: SHAP Summary Handling
    if num_dims == 3:
        n_samples, n_features, n_classes = shape
        class_index = 1 if n_classes == 2 else st_container.selectbox("Select class to visualize", range(n_classes))
        vals = values[:, :, class_index]

        if vals.shape[1] != data.shape[1]:
            vals = vals[:, :-1]

        expl = shap.Explanation(
            values=vals,
            base_values=base_values[:, class_index] if base_values.ndim == 2 else base_values,
            data=data,
            feature_names=feature_names
        )

        label = f"SHAP Summary for class {class_index}"

    elif num_dims == 2:
        if values.shape[1] != data.shape[1]:
            values = values[:, :-1]

        expl = shap.Explanation(
            values=values,
            base_values=base_values,
            data=data,
            feature_names=feature_names
        )

        label = "SHAP Summary for binary classification"

    else:
        st_container.error(f"❌ Unexpected SHAP value shape: {shape}")
        return None, "SHAP error"

    # Step 5: Generate beeswarm figure
    fig, ax = plt.subplots()
    shap.plots.beeswarm(expl, show=False)
    return fig, label

def check_required_state(keys, page_name="this step"):
    missing = [k for k in keys if k not in st.session_state or st.session_state[k] is None]
    if missing:
        st.warning(f"⚠️ Missing required step(s): `{', '.join(missing)}`.\nPlease complete them before accessing **{page_name}**.")
        st.stop()