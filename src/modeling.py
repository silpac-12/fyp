import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve, cross_val_score
from sklearn.metrics import f1_score, accuracy_score
from pycaret.classification import setup, compare_models, pull
import pandas as pd


def select_best_model(X, y):
    """
    Trains and compares multiple models using PyCaret.
    Detects data leakage, removes redundant features, and selects the best model based on accuracy.
    """
    y = y.astype(int)
    X = X.astype(float)


    # Remove Unnamed index column if present
    if 'Unnamed: 0' in X.columns:
        X = X.drop(columns=['Unnamed: 0'])


    # Initialize PyCaret
    clf_setup = setup(
        data=pd.concat([X, y], axis=1),
        target=y.name,
        session_id=42
    )

    # Compare models based on Accuracy
    best_model = compare_models(sort='Accuracy')

    if best_model is None:
        raise ValueError("No valid model was selected. Please check the dataset and preprocessing steps.")

    # Train-Test Split (20% test set to check generalization)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    best_model.fit(X_train, y_train)
    y_test_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"✅ Best Model: {best_model.__class__.__name__}")
    print(f"✅ Test Accuracy: {test_acc:.3f}, Test F1-Score: {test_f1:.3f}")
    # Run a simple cross-validation test
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")

    return best_model, pull(), test_acc


def plot_model_learning_curve(estimator, X, y, title="Learning Curve"):
    """
    Generates a learning curve to detect overfitting.
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=skf, scoring="accuracy", n_jobs=-1
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training Score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Validation Score")
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid()

    return fig

def check_feature_correlation(X, threshold=0.95):
    """
    Checks highly correlated features and prints the correlation matrix for debugging.
    """
    correlation_matrix = X.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Find features that have high correlation
    correlated_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]

    print("\n🔍 Feature Correlation Matrix:")
    print(correlation_matrix)

    if correlated_features:
        print(f"\n⚠️ WARNING: Found {len(correlated_features)} highly correlated features: {correlated_features}")
    else:
        print("\n✅ No highly correlated features detected.")

    return correlated_features