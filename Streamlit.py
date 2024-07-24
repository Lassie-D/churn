import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer


# Assuming 'churn_data' is your DataFrame
def encode_features(df): 
    df = df[df['PhoneService'] != 'No phone service']
    df['PhoneService'] = df['PhoneService'].replace({'Yes': 1, 'No': 0})
    for column in df.select_dtypes(include=['object']).columns:
        if df[column].nunique() <= 2:
            df[column] = LabelEncoder().fit_transform(df[column])
        else:
            df = pd.get_dummies(df, columns=[column], drop_first=True)
    return df


# Load and preprocess the dataset
@st.cache
def load_data():
    churn_data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # Handle 'No phone service' in all relevant columns
    churn_data.replace('No phone service', 'No', inplace=True)
    # Convert 'TotalCharges' from string to numeric, forcing errors to NaN
    churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
    # Debugging line: Check if there are any non-numeric values left
    print(churn_data[churn_data['TotalCharges'].isna()])
    # Drop NaN values that might have been created during conversion or were initially present
    churn_data.dropna(inplace=True)
    # Encode binary categories using LabelEncoder
    for column in churn_data.select_dtypes(include=['object']).columns:
        if churn_data[column].nunique() == 2:
            churn_data[column] = LabelEncoder().fit_transform(churn_data[column])
    churn_data.drop(columns=['customerID'], inplace=True)  # Drop the customerID
    return churn_data


# Load data
churn_data = load_data()

# Streamlit interface
st.title('Customer Churn Prediction')
st.write("This application predicts the churn probability for customers of a telecom company.")

# User inputs
tenure = st.slider('Tenure (months)', int(churn_data['tenure'].min()), int(churn_data['tenure'].max()))
MonthlyCharges = st.slider('Monthly Charges', float(churn_data['MonthlyCharges'].min()), float(churn_data['MonthlyCharges'].max()))
Contract = st.selectbox('Contract Type', churn_data['Contract'].unique())
InternetService = st.selectbox('Internet Service', churn_data['InternetService'].unique())
OnlineSecurity = st.selectbox('Online Security', churn_data['OnlineSecurity'].unique())

# Encode inputs
input_df = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [MonthlyCharges],
    'Contract': [Contract],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity]
})
for column in input_df.columns:
    if input_df[column].dtype == type(object):
         # Check if the column is categorical
        input_df[column] = LabelEncoder().fit_transform(input_df[column])

# Predict function
def predict_churn(input_data):
    model = LogisticRegression()
    st.write("DataFrame before fitting the model:", churn_data.head())  # Debugging line
    model.fit(churn_data.drop('Churn', axis=1), churn_data['Churn'])
    return model.predict_proba(input_data)[0][1]

# Show prediction button
if st.button('Predict Churn'):
    prediction = predict_churn(input_df)
    st.write(f'The predicted probability of churn is {prediction:.2f}')

# Additional Plots
if st.checkbox('Show data distribution'):
    st.subheader('Distribution of Monthly Charges')
    fig, ax = plt.subplots()
    sns.histplot(churn_data['MonthlyCharges'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
