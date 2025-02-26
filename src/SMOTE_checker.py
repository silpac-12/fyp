import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def check_smote_applicability(dataset, target_column=None, threshold=0.2):
    """
    Checks class imbalance and visually informs if SMOTE is applicable.
    Returns:
    - fig: Matplotlib figure object or None if an error occurs.
    - result: Dictionary with class distribution, imbalance ratio, and recommendation.
    """

    # Auto-detect target column if not provided
    if target_column is None:
        possible_targets = dataset.select_dtypes(include=['object', 'category', 'int']).columns
        target_column = possible_targets[-1] if len(possible_targets) > 0 else None

        if target_column is None:
            st.error("🚫 No suitable target column found.")
            return None, None

    # Check if target column exists in dataset
    if target_column not in dataset.columns:
        st.error(f"🚫 Target column '{target_column}' not found in dataset.")
        return None, None

    # Calculate class distribution
    class_counts = dataset[target_column].value_counts()
    if class_counts.empty:
        st.error("🚫 Target column has no valid classes.")
        return None, None

    class_distribution = {str(k): int(v) for k, v in class_counts.items()}

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 5))
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title(f"Class Distribution for '{target_column}'")
    ax.set_xlabel("Class")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticklabels(class_counts.index, rotation=0)

    # Calculate imbalance ratio
    imbalance_ratio = class_counts.min() / class_counts.max()
    recommendation = (
        f"✅ SMOTE is recommended (Imbalance ratio: {imbalance_ratio:.3f})"
        if imbalance_ratio < threshold
        else f"✔️ Class distribution is fairly balanced (Imbalance ratio: {imbalance_ratio:.3f})"
    )

    result = {
        "class_distribution": class_distribution,
        "imbalance_ratio": round(imbalance_ratio, 3),
        "recommendation": recommendation
    }

    return fig, result
