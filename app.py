import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('churn_model.h5')
# Load the encoders and scaler
with open('label_encoder.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)
with open('onehot_encoder.pkl', 'rb') as file:
    onehot_encoder_geo=pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)

#streamlit user input
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
#prepare the data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = pd.DataFrame(onehot_encoder_geo.transform([[geography]]).toarray(), 
                          columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data, geo_encoded], axis=1)

# Scale the features
input_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)
churn_probability = prediction[0][0]

if churn_probability > 0.5:
    st.error('This customer is likely to churn!')
else:
    st.success('This customer is likely to stay!')