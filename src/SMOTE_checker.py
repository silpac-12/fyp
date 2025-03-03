import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from imblearn.over_sampling import SMOTE

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


def evaluate_sampling_methods(X, y, method):
    y = y.astype(int)  # Ensure target variable is integer (0/1)
    X = X.astype(np.float64)

    # ✅ Print unique values in target variable (to check class balance)
    print("Unique values in target variable (y):", np.unique(y, return_counts=True))

    # Split into train/test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize Logistic Regression Model with regularization
    clf = LogisticRegression(max_iter=1000, C=0.1)  # C=0.1 for regularization

    # ✅ Train model on original dataset and evaluate on test set
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    base_metrics = {
        "F1 Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_pred),
    }

    # ✅ Apply SMOTE (Oversampling)
    smote = SMOTE()
    X_sm, y_sm = smote.fit_resample(X_train, y_train)

    clf.fit(X_sm, y_sm)
    y_pred_sm = clf.predict(X_test)  # Evaluate on the same test set

    smote_metrics = {
        "F1 Score": f1_score(y_test, y_pred_sm),
        "Precision": precision_score(y_test, y_pred_sm),
        "Recall": recall_score(y_test, y_pred_sm),
        "Accuracy": accuracy_score(y_test, y_pred_sm),
        "AUC-ROC": roc_auc_score(y_test, y_pred_sm),
    }

    # ✅ Apply Random Undersampling
    under = RandomUnderSampler()
    X_under, y_under = under.fit_resample(X_train, y_train)

    clf.fit(X_under, y_under)
    y_pred_under = clf.predict(X_test)  # Evaluate on the same test set

    under_metrics = {
        "F1 Score": f1_score(y_test, y_pred_under),
        "Precision": precision_score(y_test, y_pred_under),
        "Recall": recall_score(y_test, y_pred_under),
        "Accuracy": accuracy_score(y_test, y_pred_under),
        "AUC-ROC": roc_auc_score(y_test, y_pred_under),
    }

    print("Class Distribution After SMOTE:", np.bincount(y_sm))
    print("Class Distribution After Undersampling:", np.bincount(y_under))

    # ✅ Generate Recommendations Based on Results
    advice = []

    if smote_metrics["F1 Score"] > base_metrics["F1 Score"]:
        advice.append("✅ **SMOTE improved F1-score**, indicating that oversampling helped balance the model.")
    else:
        advice.append("⚠️ **SMOTE did not improve performance** significantly. Consider tuning the SMOTE parameters.")

    if under_metrics["F1 Score"] > base_metrics["F1 Score"]:
        advice.append(
            "✅ **Undersampling improved F1-score**, meaning that removing majority class examples helped reduce bias.")
    else:
        advice.append(
            "⚠️ **Undersampling did not improve performance**. Consider using a combination of SMOTE and undersampling.")

    if smote_metrics["Recall"] > base_metrics["Recall"]:
        advice.append("🚀 **SMOTE increased recall**, which is useful if you want to detect more positive cases.")

    if under_metrics["Precision"] > base_metrics["Precision"]:
        advice.append("⚖️ **Undersampling improved precision**, meaning fewer false positives.")

    return {
        "Baseline Metrics": base_metrics,
        "SMOTE Metrics": smote_metrics,
        "Undersampling Metrics": under_metrics,
        "Advice": advice
    }, clf


@st.fragment
def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, scoring="f1")

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Validation score")
    plt.xlabel("Training Size")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    return plt


def apply_smote(X, y):
    y = y.astype(int)  # Ensure target variable is integer (0/1)
    X = X.astype(np.float64)
    """Applies SMOTE to the training dataset only and returns the full dataset after processing."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE only to the training set
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # Combine train and test sets back into a DataFrame
    df_train = pd.concat([X_train_sm, y_train_sm], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    return pd.concat([df_train, df_test], axis=0).reset_index(drop=True)


def apply_undersampling(X, y):
    y = y.astype(int)  # Ensure target variable is integer (0/1)
    X = X.astype(np.float64)
    """Applies Random Undersampling to the training dataset only and returns the full dataset after processing."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply Random Undersampling only to the training set
    undersampler = RandomUnderSampler(random_state=42)
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

    # Combine train and test sets back into a DataFrame
    df_train = pd.concat([X_train_under, y_train_under], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    return pd.concat([df_train, df_test], axis=0).reset_index(drop=True)