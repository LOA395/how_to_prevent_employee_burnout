# Employee Attrition Prediction using Machine Learning

## Table of Contents
- Project Overview
- Dataset
- Project Objectives
- Key Features
- Models and Techniques
- Results


## Project Overview

Employee attrition, or turnover, is a critical challenge for businesses across industries. This project aims to predict employee attrition using various machine learning techniques. By analyzing a variety of factors, such as employee demographics, job satisfaction, and performance, this model can help HR departments proactively manage employee retention.

The goal is to build an efficient model that not only predicts whether an employee is likely to leave or stay, but also provides insights into the most important factors driving attrition.

## Dataset

The dataset includes a variety of features related to employee demographics, job characteristics, and satisfaction levels. The features include:
- **Personal attributes**: Age, gender, marital status
- **Job-related features**: Job role, monthly income, years at company, job satisfaction, etc.
- **Work environment**: Overtime, remote work, leadership opportunities

## Project Objectives

1. **Exploratory Data Analysis (EDA)**: Explore the dataset to understand the distributions, relationships, and potential outliers.
2. **Preprocessing**: Clean the data, remove outliers, handle missing values, and encode categorical variables.
3. **Feature Selection**: Analyze the impact of different features on attrition predictions to simplify the model.
4. **Modeling**: Build and evaluate multiple machine learning models to predict employee attrition.
5. **Hyperparameter Tuning**: Optimize model performance using `GridSearchCV` for hyperparameter tuning.


## Key Features

- **Data Cleaning**: Handles missing values and removes outliers using z-score.
- **Feature Engineering**: Converts categorical variables such as Yes/No to numerical equivalents and selects the most relevant features.
- **Multiple Machine Learning Models**: Evaluates several algorithms, including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
- **Hyperparameter Tuning**: Uses `GridSearchCV` to find the best hyperparameters for improved model performance.
- **Model Evaluation**: Compares models using accuracy, precision, recall, F1-score, and other relevant metrics.

## Models and Techniques

The following machine learning models were used:
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **XGBoost Classifier**

Techniques used:
- **Data Scaling**: StandardScaler and MinMaxScaler for scaling the features (optional).
- **Outlier Removal**: Z-score method for detecting and removing outliers.
- **Hyperparameter Tuning**: `GridSearchCV` for optimizing model hyperparameters.

## Results

- The best model achieved an accuracy of **77%** using Gradient Boosting with optimized hyperparameters.
- Important features contributing to employee attrition include **Job Level, Work Life Balance, Remote Work and Marital Status**.
- Surprisingly, the variables of Job Satisfaction and Monthly Income had a very low correlation with Atrittion and excluding them improved the results of the model.

## # Resource

- [Presentación](https://www.canva.com/design/DAGTF-aqEWI/eHIGmxU6ob62-Myu18UHrw/edit?utm_content=DA[…]m_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Autores:
  - `Laura Ortiz Alameda` - [Linkedin](https://www.linkedin.com/in/laura-ortiz-alameda/)
  - `Almudena Ocaña López-Gasco` - [Linkedin](https://www.linkedin.com/in/almudena-ocaloga/)
