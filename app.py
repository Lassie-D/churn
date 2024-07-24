import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load your trained logistic regression model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to encode the inputs using the same label encoders used in training
def encode_inputs(input_data):
    label_encoders = {}
    categorical_columns = input_data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        input_data[column] = le.fit_transform(input_data[column])
        label_encoders[column] = le
    return input_data

# Streamlit user interface
def user_input_features():
    gender = st.selectbox('Gender', ('Male', 'Female'))
    SeniorCitizen = st.selectbox('Senior Citizen', (0, 1))
    Partner = st.selectbox('Partner', ('Yes', 'No'))
    Dependents = st.selectbox('Dependents', ('Yes', 'No'))
    tenure = st.slider('Tenure', 0, 72, 1)  # Example for numerical input
    MonthlyCharges = st.slider('Monthly Charges', 0, 200, 1)
    TotalCharges = st.slider('Total Charges', 0, 8000, 1)
    data = {'gender': [gender],
            'SeniorCitizen': [SeniorCitizen],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'tenure': [tenure],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges]}
    features = pd.DataFrame(data)
    return features

st.write("""
# Simple Customer Churn Prediction App
This app predicts the probability of a telecom customer churning.
""")

input_df = user_input_features()
# Encode the input variables
encoded_input_df = encode_inputs(input_df)
# Display user input features
st.subheader('User Input features')
st.write(encoded_input_df)
# Predict and display the output
if st.button('Predict'):
    prediction = model.predict(encoded_input_df)
    prediction_proba = model.predict_proba(encoded_input_df)
    st.subheader('Prediction')
    st.write('Churn' if prediction[0] == 1 else 'No churn')
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
