# Libraries
import streamlit as st
import pickle
import numpy as np


# Load the model from the pickle file
with open('models/final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit title and description
st.title('Employee Attrition Prediction')
st.write('Fill in the employee details below to predict whether the employee will leave or stay.')

# Step 1: Personal Information
st.header("1. Personal Information")
age = st.number_input('Age', min_value=18, max_value=65, value=30)
gender_input = st.selectbox('Gender', ['Female', 'Male'])
gender = 1 if gender_input == 'Female' else 0  # 1 for Female, 0 for Male

marital_status_input = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
marital_status = 1 if marital_status_input == 'Single' else 0  # 1 for Single, 0 for others

number_of_dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, value=2)

# Step 2: Employment Details
st.header("2. Employment Details")
years_at_company = st.number_input('Years at Company', min_value=0, max_value=50, value=5)
number_of_promotions = st.number_input('Number of Promotions', min_value=0, max_value=10, value=1)
distance_from_home = st.number_input('Distance from Home (in miles)', min_value=0, max_value=100, value=10)

job_level_input = st.selectbox('Job Level', ['Entry', 'Mid', 'Senior'])
job_level = {'Entry': 0, 'Mid': 1, 'Senior': 2}[job_level_input]

education_level_input = st.selectbox('Education Level', ['High School', 'Associate Degree', 'Bachelor Degree', "Master Degree", 'Doctoral Degree'])
education_level = {'High School': 0, 'Associate Degree': 1, 'Bachelor Degree': 2, 'Master Degree': 3, 'Doctoral Degree': 4}[education_level_input]

company_size_input = st.selectbox('Company Size', ['Small', 'Medium', 'Large'])
company_size = {'Small': 0, 'Medium': 1, 'Large': 2}[company_size_input]

company_tenure = st.number_input('Company Tenure (Years)', min_value=0, max_value=200, value=5)

# Step 3: Work Characteristics
st.header("3. Work Characteristics")
work_life_balance_input = st.selectbox('Work-Life Balance', ['Bad', 'Below Average', 'Good', 'Excellent'])
work_life_balance = {'Bad': 0, 'Below Average': 1, 'Good': 2, 'Excellent': 3}[work_life_balance_input]

performance_rating_input = st.selectbox('Performance Rating', ['Low', 'Below Average', 'Average', 'Good'])
performance_rating = {'Low': 0, 'Below Average': 1, 'Average': 2, 'Good': 3}[performance_rating_input]

overtime_input = st.selectbox('Overtime', ['Yes', 'No'])
overtime = 1 if overtime_input == 'Yes' else 0

remote_work_input = st.selectbox('Remote Work', ['Yes', 'No'])
remote_work = 1 if remote_work_input == 'Yes' else 0

# Step 4: Employee Perceptions
st.header("4. Employee Perceptions")
leadership_opportunities_input = st.selectbox('Leadership Opportunities', ['Yes', 'No'])
leadership_opportunities = 1 if leadership_opportunities_input == 'Yes' else 0

innovation_opportunities_input = st.selectbox('Innovation Opportunities', ['Yes', 'No'])
innovation_opportunities = 1 if innovation_opportunities_input == 'Yes' else 0

company_reputation_input = st.selectbox('Company Reputation', ['Poor', 'Below Average', 'Good', 'Excellent'])
company_reputation = {'Poor': 0, 'Below Average': 1, 'Good': 2, 'Excellent': 3}[company_reputation_input]

employee_recognition_input = st.selectbox('Employee Recognition', ['Low', 'Average', 'Good', 'High'])
employee_recognition = {'Low': 0, 'Average': 1, 'Good': 2, 'High': 3}[employee_recognition_input]

# Button to make predictions
if st.button('Predict'):
    # Input features in the same order as your model expects
    input_data = np.array([[age, years_at_company, work_life_balance, performance_rating,
                            number_of_promotions, overtime, distance_from_home, education_level,
                            number_of_dependents, job_level, company_size, company_tenure, remote_work,
                            leadership_opportunities, innovation_opportunities, company_reputation,
                            employee_recognition, gender, marital_status]])

    # Make prediction
    prediction = model.predict(input_data)

    # Display the result
    if prediction[0] == 1:
        st.error('Prediction: The employee is likely to leave.')
    else:
        st.success('Prediction: The employee is likely to stay.')