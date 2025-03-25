import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.stats import pointbiserialr


def show_summary(df):
    """Returns summary statistics of the dataset"""
    return df.describe()

def plot_histograms(df: pd.DataFrame, show_plot: bool = True):
    """
    Plots histograms for numerical columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - show_plot (bool): If True, displays plots immediately. If False, returns figure objects for external handling.

    Returns:
    - List[matplotlib.figure.Figure]: List of figure objects (useful for saving or external rendering).
    """
    figures = []
    numeric_columns = df.select_dtypes(include="number").columns

    if numeric_columns.empty:
        print("⚠️ No numerical columns found to plot histograms.")
        return figures

    for col in numeric_columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        figures.append(fig)
    return figures

def compare_feature_means(df_before, df_after):
    """Returns a DataFrame comparing feature means before and after imputation."""
    means_before = df_before.mean(numeric_only=True)
    means_after = df_after.mean(numeric_only=True)
    return pd.DataFrame({
        'Before Imputation': means_before,
        'After Imputation': means_after,
        'Difference': means_after - means_before
    })

def compare_feature_stds(df_before, df_after):
    """Returns a DataFrame comparing feature standard deviations before and after imputation."""
    std_before = df_before.std(numeric_only=True)
    std_after = df_after.std(numeric_only=True)
    return pd.DataFrame({
        'Before Imputation': std_before,
        'After Imputation': std_after,
        'Difference': std_after - std_before
    })

def compare_missing_values(df_before, df_after):
    """Returns a DataFrame comparing missing value counts before and after imputation."""
    missing_before = df_before.isnull().sum()
    missing_after = df_after.isnull().sum()
    return pd.DataFrame({
        'Before Imputation': missing_before,
        'After Imputation': missing_after,
        'Difference': missing_after - missing_before
    })

def compare_correlation_matrices(df_before, df_after):
    """Returns a matplotlib figure comparing correlation matrices before and after imputation."""
    numeric_before = df_before.select_dtypes(include=["number"])
    numeric_after = df_after.select_dtypes(include=["number"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(numeric_before.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0])
    axes[0].set_title("Correlation Matrix Before Imputation")

    sns.heatmap(numeric_after.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1])
    axes[1].set_title("Correlation Matrix After Imputation")

    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, show_plot: bool = True, debug: bool = False):
    """
    Plots a correlation heatmap using only numeric columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        show_plot (bool): Whether to display the plot (default: True).
        debug (bool): If True, prints information about excluded columns.

    Returns:
        plt.Figure: The matplotlib figure object containing the heatmap.
    """

    # Identify non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=["number"]).columns.tolist()

    if debug:
        print(f"🔍 Excluded non-numeric columns: {non_numeric_columns}")

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        if debug:
            print("❌ No numeric columns available for correlation matrix.")
        return None

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")

    return fig


def detect_data_leakage(X, y, threshold=0.85):
    """
    Detects features highly correlated with the binary target variable y.

    :param X: Feature DataFrame
    :param y: Target Series (binary: 0 or 1)
    :param threshold: Correlation threshold for potential leakage (default: 0.85)

    :return: Dictionary containing features causing data leakage with explanations
    """

    # Ensure y is a pandas Series and has no missing values
    y = pd.Series(y).dropna()

    # Ensure X is numeric only
    X = X.select_dtypes(include=[np.number])

    # Compute correlation of features with the target (using point-biserial correlation)
    target_correlation = {col: pointbiserialr(X[col], y)[0] for col in X.columns}

    # Identify potential data leakage
    potential_leaks = {
        feature: f"Correlation: {corr:.3f} (Threshold: {threshold})"
        for feature, corr in target_correlation.items() if abs(corr) >= threshold
    }

    return potential_leaks  # Return as a dictionary with explanations


def detect_statistical_shifts(before_df, after_df, threshold=0.3):
    """
    Detects features with significant changes in mean or standard deviation after imputation.

    :param before_df: Original dataset before imputation.
    :param after_df: Dataset after imputation.
    :param threshold: Percentage change threshold for flagging bias (default: 30%).

    :return: DataFrame with change in mean and std deviation.
    """
    before_stats = before_df.describe().T[['mean', 'std']]
    after_stats = after_df.describe().T[['mean', 'std']]

    shifts = after_stats - before_stats
    shifts['mean_change_pct'] = (shifts['mean'] / before_stats['mean']).abs()
    shifts['std_change_pct'] = (shifts['std'] / before_stats['std']).abs()

    flagged_features = shifts[(shifts['mean_change_pct'] > threshold) | (shifts['std_change_pct'] > threshold)]

    return flagged_features[['mean_change_pct', 'std_change_pct']].reset_index()

def detect_target_correlation_shifts(before_df, after_df, y, threshold=0.2):
    """
    Detects features with a large change in correlation with the target after imputation.

    :param before_df: Feature DataFrame before imputation.
    :param after_df: Feature DataFrame after imputation.
    :param y: Target variable (binary, multi-class, or continuous).
    :param threshold: Absolute change threshold for flagging features (default: 0.2).

    :return: DataFrame of features with large correlation changes.
    """

    # Ensure y is a Pandas Series
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="target")

    # Drop NaN values from y
    y = y.dropna()

    # Ensure feature dataframes contain only numeric columns
    before_df = before_df.select_dtypes(include=[np.number])
    after_df = after_df.select_dtypes(include=[np.number])

    # Zero-fill missing values in before_df
    before_df = before_df.fillna(0)

    # Determine correlation method based on y
    def compute_corr(col, y):
        if y.nunique() == 2:  # Binary classification (0/1 target)
            return pointbiserialr(col, y)[0] if col.nunique() > 1 else 0
        else:  # Multi-class or continuous target
            return col.corr(y) if col.nunique() > 1 else 0

    # Compute correlation before and after imputation
    before_corr = before_df.apply(lambda col: compute_corr(col, y), axis=0)
    after_corr = after_df.apply(lambda col: compute_corr(col, y), axis=0)

    # Compute absolute correlation differences
    corr_diff = (after_corr - before_corr).abs()

    # Select features where correlation change exceeds the threshold
    flagged_features = corr_diff[corr_diff > threshold]

    # Return results as a DataFrame
    return pd.DataFrame({
        'feature': flagged_features.index,
        'correlation_change': flagged_features.values,
        'before_correlation': before_corr[flagged_features.index].values,
        'after_correlation': after_corr[flagged_features.index].values
    }).reset_index(drop=True)





def detect_outlier_changes(before_df, after_df, method='zscore', threshold=3):
    """
    Detects whether imputation has introduced or removed a significant number of outliers.

    :param before_df: Data before imputation.
    :param after_df: Data after imputation.
    :param method: Method for detecting outliers ('zscore' or 'iqr').
    :param threshold: Z-score or IQR threshold for outliers.

    :return: DataFrame showing change in outlier count for each feature.
    """

    def count_outliers(df, method, threshold):
        if method == 'zscore':
            return ((df - df.mean()) / df.std()).abs() > threshold
        elif method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            return (df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))

    before_outliers = count_outliers(before_df, method, threshold).sum()
    after_outliers = count_outliers(after_df, method, threshold).sum()

    outlier_diff = after_outliers - before_outliers
    return pd.DataFrame({'feature': outlier_diff.index, 'outlier_count_change': outlier_diff.values})
