# generate_prompt.py
import pandas as pd


def eda_prompt(missing_summary, corr_mat):
    return f"""
The clinician uploaded a medical dataset for model training.

**Missing Data Summary:** {missing_summary}

**Correlation Matrix:** {corr_mat}

Explain these issues in simple medical terms, suggest potential problems during modeling, give detailed feedback on relevant values, and how the clinician can interpret them.
"""

def shap_prompt(top_features):
    return f"""
The model's predictions were explained using SHAP (SHapley Additive exPlanations).

Top features influencing predictions: {', '.join(top_features)}.

Explain how these features impact the model and what the clinician should be aware of when interpreting results.
"""

def imputation_prompt(
    imputation_method: str,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    dropped_cols: list,
    corr_before: pd.DataFrame,
    corr_after: pd.DataFrame
) -> str:
    # --- Missing Summary ---
    missing_cols = df_before.columns[df_before.isnull().any()].tolist()
    num_missing = df_before[missing_cols].isnull().sum().to_dict()
    missing_summary = "\n".join([f"- {col}: {count} missing" for col, count in num_missing.items()])

    # --- Dropped Columns ---
    dropped_summary = ", ".join(dropped_cols) if dropped_cols else "None"

    # --- Feature Mean and Std Shifts ---
    shifted_means = (df_before.mean() - df_after.mean()).abs().sort_values(ascending=False)
    mean_change_summary = "\n".join([f"- {k}: Δ = {v:.4f}" for k, v in shifted_means.head(5).to_dict().items()])

    shifted_stds = (df_before.std() - df_after.std()).abs().sort_values(ascending=False)
    std_change_summary = "\n".join([f"- {k}: Δ = {v:.4f}" for k, v in shifted_stds.head(5).to_dict().items()])

    # --- Outlier-Like Changes ---
    outlier_counts = (df_after - df_before).abs().gt(3 * df_before.std()).sum()
    outlier_summary = "\n".join([f"- {k}: {v} large shifts" for k, v in outlier_counts[outlier_counts > 0].head(5).to_dict().items()])

    # --- Correlation Matrix Change ---
    corr_diff = (corr_before - corr_after).abs()
    # Unstack and sort to find top pairwise correlation changes
    top_corr_changes = corr_diff.where(~corr_diff.isna()).unstack().sort_values(ascending=False)
    # Remove duplicate entries (like (A,B) vs (B,A))
    top_corr_changes = top_corr_changes[top_corr_changes.index.get_level_values(0) < top_corr_changes.index.get_level_values(1)]
    top_corr_summary = "\n".join([
        f"- {a} vs {b}: Δ = {delta:.4f}"
        for (a, b), delta in top_corr_changes.head(5).items()
    ])

    # --- Final Prompt ---
    return f"""
A medical dataset was processed to prepare it for machine learning-based cancer prediction. Imputation was used to handle missing data.

🩺 **Imputation Method Used:** {imputation_method.upper()}
🗂️ **Columns Removed Before Imputation:** {dropped_summary}

📉 **Missing Values Summary:**
{missing_summary or 'None'}

📈 **Top Feature Mean Shifts:**
{mean_change_summary or 'None'}

📊 **Top Std. Deviation Shifts:**
{std_change_summary or 'None'}

⚠️ **Outlier-Like Feature Shifts:**
{outlier_summary or 'None'}

🔄 **Top Correlation Changes Between Feature Pairs:**
{top_corr_summary or 'None'}

🎯 **Instructions for ChatGPT:**
Please explain the clinical implications of the imputation process and the changes in data distribution and correlation:
- Are the observed changes acceptable?
- Do any correlation shifts raise red flags about clinical reliability?
- Would these transformations affect how doctors interpret feature relationships?
"""

def modeling_prompt(
    model_name: str,
    model_params: dict,
    test_accuracy: float,
    cv_scores: list,
    precision: float,
    recall: float,
    shap_summary: dict,
    top_feature_corrs: dict,
    correlated_features: list,
    n_classes: int
) -> str:
    # Format top SHAP features
    shap_features = sorted(shap_summary.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    shap_summary_text = "\n".join([f"- {feat}: SHAP impact score = {score:.4f}" for feat, score in shap_features])

    # Format top target correlations
    top_corrs_text = "\n".join([f"- {feat}: correlation with target = {score:.4f}" for feat, score in list(top_feature_corrs.items())[:5]])

    # Correlated feature pairs removed or flagged
    correlated_text = ", ".join(correlated_features) if correlated_features else "None"

    # CV score summary
    mean_cv = sum(cv_scores) / len(cv_scores)
    cv_summary = f"Mean: {mean_cv:.4f}, Range: {min(cv_scores):.4f}–{max(cv_scores):.4f}"

    # Format model params (optional short list)
    param_summary = ", ".join([f"{k}={v}" for k, v in list(model_params.items())[:5]])

    # Build prompt
    return f"""
A machine learning model was trained to predict a medical diagnosis based on patient data.

🧠 **Selected Model:** {model_name}
🔧 **Key Parameters:** {param_summary}

📊 **Evaluation Metrics:**
- Test Accuracy: {test_accuracy:.3f}
- Cross-Validation Accuracy: {cv_summary}
- Precision: {precision:.3f}
- Recall (Sensitivity): {recall:.3f}
- Classification Type: {"Binary" if n_classes == 2 else f"{n_classes}-Class"}

🔍 **Top SHAP Features (most impactful for predictions):**
{shap_summary_text or 'N/A'}

📈 **Top Feature Correlations with Target:**
{top_corrs_text or 'N/A'}

🚨 **Highly Correlated Feature Pairs Detected (potential multicollinearity):**
{correlated_text or 'None'}

🎯 **Instructions for ChatGPT:**
- Please explain whether the model appears clinically reliable.
- Does the precision and recall suggest underdiagnosis or overdiagnosis risk?
- Do the SHAP features align with clinical expectations?
- Are there any red flags in feature correlation or collinearity that might bias decisions?
- How should a clinician interpret predictions from this model?
"""
