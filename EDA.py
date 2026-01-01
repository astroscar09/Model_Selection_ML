import pandas as pd
import numpy as np
from plotting_utils import plot_histograms_from_df
import seaborn as sns
import matplotlib.pyplot as plt

EDA_DIR = 'EDA'

def corr_matrix(df):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.savefig(f'{EDA_DIR}/Correlation_Matrix_Plots.pdf')
    #plt.show()

def df_overview(df):
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nUnique Values:\n", df.nunique())

def df_stats(df):
    numeric = df.select_dtypes(include='number')
    print(numeric.describe())
    print("\nSkewness:\n", numeric.skew())
    print("\nKurtosis:\n", numeric.kurt())

def plot_box(df):
    numeric = df.select_dtypes(include='number')
    df[numeric.columns].plot(kind='box', figsize=(12,6), vert=False)
    plt.savefig(f'{EDA_DIR}/Box_Plots.pdf')

def scatter_plot(df, col1, col2):
    sns.scatterplot(data=df, x=col1, y=col2)
    plt.savefig(f'{EDA_DIR}/Scatter_Plots_{col1}_{col2}.pdf')

def pair_plot(df):
    sns.pairplot(df.select_dtypes(include='number'))
    plt.savefig(f'{EDA_DIR}/Pair_Plot.pdf')

def missing_info(df):
    return (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

def eda_report(df, target=None):
    print("=== Data Overview ===")
    df_overview(df)
    print("\n=== Missing Values ===")
    print(missing_info(df))
    
    corr_matrix(df)
    plot_box(df)
    fig, ax = plot_histograms_from_df(df)
    fig.savefig(f'{EDA_DIR}/Histogram_Plots.pdf')


