# Libraries
import os
import sys
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier


# Functions
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



# load data
data = pd.read_csv('data/raw/hr_test.csv')

# Standardize the column names
data.columns = data.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")

# drop the employee id column
data.drop(columns=['employee_id'], inplace=True)


#Define columns types
# Binary columns (Yes/No) that need to be converted into numerical format (1 for Yes, 0 for No).
yes_no_cols = ['overtime', 'remote_work', 'leadership_opportunities', 'innovation_opportunities']

# Categorical columns where the categories have no inherent order
nominal_cols = ['gender', 'job_role', 'marital_status']

# Categorical columns where the categories follow a specific order
ordinal_cols = ['work_life_balance', 'job_satisfaction', 'performance_rating', 'education_level', 'job_level',
                'company_size', 'company_reputation', 'employee_recognition' ]

# Continuous or discrete numerical columns
numerical_cols =['age', 'years_at_company', 'monthly_income', 'number_of_promotions', 'distance_from_home',
                 'number_of_dependents', 'company_tenure']

# Target variable.
target_col = ['attrition']

# Run the function to convert Yes/No to 1/0
yes_no_to_numeric(data, yes_no_cols)

# One-hot encoding for sex and title
data = pd.get_dummies(data, columns=nominal_cols)

# Defining Category Orders
categories = [['Poor', 'Fair', 'Good', 'Excellent'],  # Work Life Balance
              ['Very Low', 'Low', 'Medium', 'High', 'Very High'], # Job Satisfaction
              ['Low', 'Below Average', 'Average', 'High'], # Performance Rating
              ['High School', 'Associate Degree', "Bachelor’s Degree", "Master’s Degree", 'PhD'], # Education Level
              ['Entry', 'Mid', 'Senior'], # Job Level          
              ['Small', 'Medium', 'Large'], # Company Size
              ['Poor', 'Fair', 'Good', 'Excellent'], # Company Reputation
              ['Very Low', 'Low', 'Medium', 'High', 'Very High'] # Employee Recognition
              ]

# Applying Ordinal Encoding
encoder = OrdinalEncoder(categories=categories)
encoded_data = encoder.fit_transform(data[ordinal_cols])

# Replacing Encoded Data
df_encoded = pd.DataFrame(encoded_data, columns=ordinal_cols)
data[ordinal_cols] = df_encoded[ordinal_cols]

# Attrition is transformed into numerical values
data['attrition'] = data['attrition'].replace({'Stayed': 0, 'Left': 1})

# Split the data into features and target
x = data.drop(columns=['attrition'])
y = data['attrition']

all_features = x.columns.tolist()
numeric_features = [col for col in ['age', 'work_life_balance', 'education_level', 'job_level', 
                                    'performance_rating', 'company_size', 'company_reputation', 'job_satisfaction', 
                                    'years_at_company', 'monthly_income', 'number_of_promotions', 'distance_from_home', 
                                    'number_of_dependents', 'company_tenure', 'employee_recognition'] if col in x.columns]

# Select the 'Without Selection' feature set 
X_reduced = x[[
    col for col in all_features if col not in ['job_satisfaction', 'monthly_income', 'job_role_Education',
                                                'job_role_Finance', 'job_role_Healthcare', 'job_role_Media',
                                                'job_role_Technology', 'marital_status_Married', 'marital_status_Divorced',]]]


# Remove outliers
X_reduced_clean = remove_outliers(X_reduced, numeric_features)
y_reduced_clean = y[X_reduced_clean.index]

# Split the data into training and testing sets
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced_clean, y_reduced_clean, test_size=0.2, random_state=42)


# Initialize the Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42, learning_rate = 0.1, max_depth = 3, max_features ='sqrt', min_samples_leaf = 1, min_samples_split =  10, n_estimators = 200, subsample = 0.6)

# Train the model on raw (unscaled) data
gb_model.fit(X_train_reduced, y_train_reduced)


# Save the model as a pickle file
filename = 'models/final_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(gb_model, file)
