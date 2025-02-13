import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def show_summary(df):
    """Returns summary statistics of the dataset"""
    return df.describe()

def plot_histograms(df):
    """Plots histograms for numerical columns"""
    for col in df.select_dtypes(include="number").columns:
        plt.figure()
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()


def plot_correlation_heatmap(df):
    """Plots correlation heatmap after handling non-numeric data"""

    # Convert date columns to datetime
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass  # Ignore columns that cannot be converted to datetime

    # Drop non-numeric columns (e.g., categorical, datetime, text)
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        print("No numeric columns available for correlation matrix.")
        return

    # Plot correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
