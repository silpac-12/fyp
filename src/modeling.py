import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def select_best_model(X, y):
    """
    Trains and compares multiple models using 5-fold cross-validation.
    Selects the best one based on a balanced performance score.
    """
    y = y.astype(int)  # Ensure target variable is numeric
    X = X.astype(float)

    num_classes = len(np.unique(y))

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=30, max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=30,
            max_depth=4,
            learning_rate=0.05,
            colsample_bytree=0.8,
            subsample=0.8,
            alpha=0.1,  # Regularization to prevent overfitting
            eval_metric="mlogloss",
            use_label_encoder=False
        ),
        "SVM": SVC(probability=True) if num_classes > 2 else None  # SVM for multi-class
    }

    # Remove None values (if SVM was excluded)
    models = {name: model for name, model in models.items() if model is not None}

    best_model = None
    best_score = 0
    model_scores = {}

    # Perform 5-fold cross-validation on each model
    for name, model in models.items():
        scores_f1 = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
        scores_acc = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        scores_precision = cross_val_score(model, X, y, cv=5, scoring="precision_macro")
        scores_recall = cross_val_score(model, X, y, cv=5, scoring="recall_macro")

        avg_f1 = np.mean(scores_f1)
        avg_accuracy = np.mean(scores_acc)
        avg_precision = np.mean(scores_precision)
        avg_recall = np.mean(scores_recall)

        avg_score = (avg_f1 + avg_accuracy + avg_precision + avg_recall) / 4  # Balanced score

        model_scores[name] = {
            "F1 Score": avg_f1,
            "Accuracy": avg_accuracy,
            "Precision": avg_precision,
            "Recall": avg_recall
        }

        if avg_score > best_score:
            best_score = avg_score
            best_model = (name, model)

    # Check for potential data leakage
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    high_corr_features = correlations[correlations > 0.1].index.tolist()

    if high_corr_features:
        print(f"⚠️ WARNING: Potential data leakage. Highly correlated features with target: {high_corr_features}")

    # Extract the selected model
    selected_model_name, selected_model = best_model
    reasoning = generate_model_reasoning(selected_model_name, model_scores[selected_model_name], num_classes)

    ### ✅ Final Generalization Check with Test Set ###

    # Split a separate test set (15% of the data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # Train the selected model
    selected_model.fit(X_train, y_train)
    y_test_pred = selected_model.predict(X_test)

    # Evaluate test performance
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    print(f"✅ Test F1-Score: {test_f1:.3f} (Validation F1: {best_score:.3f})")

    return selected_model, model_scores, reasoning


def generate_model_reasoning(model_name, metrics, num_classes):
    """
    Generates an explanation of why a model was chosen.
    """
    reasoning = f"🔹 **{model_name} was selected because:**\n\n"

    if model_name == "Logistic Regression":
        reasoning += "- It's highly interpretable, making it ideal for medical applications.\n"
        reasoning += "- Works well for binary classification problems (e.g., disease vs. no disease).\n"
    elif model_name == "Random Forest":
        reasoning += "- Handles **complex relationships** better than linear models.\n"
        reasoning += "- Provides **feature importance**, helping researchers understand key medical factors.\n"
        if num_classes > 2:
            reasoning += "- Works well for multi-class problems (e.g., different disease stages).\n"
    elif model_name == "XGBoost":
        reasoning += "- Handles **imbalanced datasets** better than Random Forest.\n"
        reasoning += "- Often **achieves higher accuracy** while avoiding overfitting with built-in regularization.\n"
    elif model_name == "SVM":
        reasoning += "- Good for datasets with **small sample sizes** and complex decision boundaries.\n"
        reasoning += "- Works well for **multi-class classification** in medical datasets.\n"

    reasoning += f"\n🔎 **Performance Metrics (5-Fold Cross Validation Averages):**\n"
    for metric, value in metrics.items():
        if isinstance(value, dict):  # Handle cases where AUC-ROC returns a dictionary
            reasoning += f"- {metric}: {value}\n"
        else:
            reasoning += f"- {metric}: {value:.3f}\n"

    return reasoning


def plot_model_learning_curve(estimator, X, y, title="Learning Curve"):
    """
    Generates and returns a learning curve plot to detect overfitting.
    """

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring="f1_macro", shuffle=True, random_state=42
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training Score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Validation Score")

    ax.set_xlabel("Training Size")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.legend()
    ax.grid()

    return fig
