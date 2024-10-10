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

# Input fields for all the model variables
age = st.number_input('Age', min_value=18, max_value=65, value=30)
years_at_company = st.number_input('Years at Company', min_value=0, max_value=40, value=5)
work_life_balance = st.selectbox('Work-Life Balance (1=Bad, 4=Excellent)', [1, 2, 3, 4])
performance_rating = st.selectbox('Performance Rating (1=Low, 5=High)', [1, 2, 3, 4, 5])
number_of_promotions = st.number_input('Number of Promotions', min_value=0, max_value=10, value=1)
overtime = st.selectbox('Overtime (Yes=1, No=0)', [1, 0])
distance_from_home = st.number_input('Distance from Home (in miles)', min_value=0, max_value=100, value=10)
education_level = st.selectbox('Education Level (1=High School, 2=Bachelor, 3=Master, 4=PhD)', [1, 2, 3, 4])
number_of_dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, value=2)
job_level = st.selectbox('Job Level (1=Entry, 5=Executive)', [1, 2, 3, 4, 5])
company_size = st.selectbox('Company Size (1=Small, 3=Large)', [1, 2, 3])
company_tenure = st.number_input('Company Tenure (Years)', min_value=0, max_value=40, value=5)
remote_work = st.selectbox('Remote Work (Yes=1, No=0)', [1, 0])
leadership_opportunities = st.selectbox('Leadership Opportunities (Yes=1, No=0)', [1, 0])
innovation_opportunities = st.selectbox('Innovation Opportunities (Yes=1, No=0)', [1, 0])
company_reputation = st.selectbox('Company Reputation (1=Poor, 4=Excellent)', [1, 2, 3, 4])
employee_recognition = st.selectbox('Employee Recognition (1=Low, 5=High)', [1, 2, 3, 4, 5])
gender = st.selectbox('Gender (Female=1, Male=0)', [1, 0])
marital_status = st.selectbox('Marital Status (Single=1)', [1])

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