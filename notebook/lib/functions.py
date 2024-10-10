# Libraries

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  
from scipy import stats  
from scipy.stats import chi2_contingency  
from scipy.stats.contingency import association


# data exploration function
def df_exploration(df):
    """
    Perform basic exploratory analysis on the dataframe.

    Parameters:
    df: pandas DataFrame
        The DataFrame to explore.

    Returns:
    df: pandas DataFrame
        The original DataFrame, unchanged.
    """
    # dataframe information 
    print(df.info())  # check for data types
    print(f"\nValores duplicados: {df.duplicated().sum()}")  # check if there are any duplicated values
    print(f"\nValores nulos: \n{df.isnull().sum()}") # check if there are any null values
    print (f"\nValores unicos: \n{df.nunique()}") # check for unique values
    return df

# Function to Get Value Counts for Multiple Columns
def get_value_counts(df, columns):
    """
    Get the value counts for a list of columns in a DataFrame.

    Parameters:
    df: pandas DataFrame
        The DataFrame containing the data.
    columns: list
        List of column names for which to get value counts.

    Returns:
    value_counts_dict: dict
        A dictionary where keys are column names and values are strings with value counts.
    """
    value_counts_dict = {}
    for column in columns:
        value_counts_dict[column] = f"\n{column}:\n{df[column].value_counts().to_string()}\n" # Converts the value counts to a string format for easier display
    return value_counts_dict

# Categorical univariate analysis function
def eda_uni_cat(df, col):
    """
    Perform univariate analysis for a categorical variable.

    Parameters:
    df: pandas DataFrame
        The DataFrame containing the data.
    col: str
        The name of the categorical column to analyze.

    Returns:
    freq_table: pandas DataFrame
        A DataFrame with the frequency distribution of the column.
    """
    # Create the frequency table with percentages
    freq_table = pd.DataFrame({'Frecuencia relativa (%)': df[col].value_counts(normalize=True) * 100})
    print(freq_table)

    # Visualize the frequency in a countplot
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col)
    plt.title(f'Gráfico de frecuencia para la columna: {col}')
    plt.ylabel('Frecuencia absoluta')
    plt.xlabel(col.capitalize())
    plt.xticks(rotation=90)
    plt.show()

    return freq_table

# Categorical bivariate analysis function
def eda_bi_cat(df, cat_col, target_cat_col):
    """
    Perform bivariate analysis for two categorical variables.

    Parameters:
    df: pandas DataFrame
        The DataFrame containing the data.
    cat_col: str
        The name of the primary categorical column to analyze.
    target_cat_col: str
        The target categorical column (e.g., 'attrition').

    Returns:
    crosstab: pandas DataFrame
        A DataFrame showing the percentage distribution of the target column by the primary categorical column.
    """
    #Create a crosstab with percentages
    crosstab = pd.crosstab(df[cat_col], df[target_cat_col], normalize='index') * 100

    #Calculate the P-value and Cramer's value
    chi2, p, _, _ = chi2_contingency(pd.crosstab(df[cat_col], df[target_cat_col]))

    cramer_v = association((pd.crosstab(df[cat_col], df[target_cat_col])), method="cramer")

    #Print the results
    print(f"p-value de Chi-cuadrado: {p:.4f}")
    print(f"Cramér's V: {cramer_v:.4f}\n")

    # Visualize the results with a barplot with percentages
    crosstab_plot = crosstab.reset_index().melt(id_vars=[cat_col], var_name=target_cat_col, value_name='percentage')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=crosstab_plot, x=cat_col, y='percentage', hue=target_cat_col, palette='viridis')
    plt.title(f'Percentage of {target_cat_col} by {cat_col}')
    plt.ylabel('Percentage (%)')
    plt.xlabel(cat_col.capitalize())
    plt.xticks(rotation=90)
    plt.legend(title=target_cat_col)
    plt.show()


    return crosstab

# Numerical univariable and bivariable analysis function
def eda_numerical_analysis(df, numerical_columns, hue_column='attrition'):
    """
    Perform univariate and bivariate analysis for numerical columns.

    Parameters:
    df: pandas DataFrame
        The DataFrame containing the data.
    numerical_columns: list
        List of numerical columns to analyze.
    hue_column: str, optional (default='attrition')
        The categorical target column to compare against in bivariate analysis.

    Returns:
    None
    """
    for column in numerical_columns:
        # Análisis univariable
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column], kde=True, bins=30, color='blue')
        plt.title(f'Distribución de {column}')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        plt.show()

        # Análisis bivariable respecto a 'attrition'
        plt.figure(figsize=(10, 5))
        sns.kdeplot(data=df, x=column, hue=hue_column, fill=True, palette='viridis')
        plt.title(f'Distribución de {column} respecto a si el empleado abandonó')
        plt.xlabel(column)
        plt.ylabel('Densidad')
        plt.legend(title=hue_column, labels=['Abandonó', 'No abandonó'])
        plt.show()

        # Boxplot de las variables numéricas respecto a 'attrition'
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x=hue_column, y=column, palette='viridis')
        plt.title(f'Boxplot de {column} respecto a si el empleado abandonó')
        plt.xlabel('Abandono (attrition)')
        plt.ylabel(column)
        plt.xticks([0, 1], ['No abandonó', 'Abandonó'])
        plt.show()

# Function to convert Yes/No to 1/0
def yes_no_to_numeric(df, list):
    """
    Convert Yes/No values to 1/0 in the specified columns.

    Parameters:
    df: pandas DataFrame
        The DataFrame containing the data.
    list: list
        List of column names where Yes/No should be converted to 1/0.

    Returns:
    df: pandas DataFrame
        The DataFrame with Yes/No values converted to 1/0.
    """
    # Boolean transformations
    yes_no_dict = {'Yes': 1, 'No': 0}
    
    for col in list:
        df[col].replace(yes_no_dict, inplace=True)
   
    return df

# Function to remove outliers
def remove_outliers(df, columns, z_threshold=3):
    """
    Remove outliers from the specified columns using the z-score method.

    Parameters:
    df: pandas DataFrame
        The DataFrame from which to remove outliers.
    columns: list
        List of numerical columns to check for outliers.
    z_threshold: int, optional (default=3)
        The z-score threshold above which a value is considered an outlier.

    Returns:
    df: pandas DataFrame
        The DataFrame with outliers removed.
    """
    columns = [col for col in columns if col in df.columns]
    return df[(np.abs(stats.zscore(df[columns])) < z_threshold).all(axis=1)]