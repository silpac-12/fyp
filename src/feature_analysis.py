import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

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