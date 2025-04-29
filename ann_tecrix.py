import streamlit as st
import numpy as np
import tensorflow as tf

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle


# Load the model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## stremlit

st.title("Customer Churn Prediction App 💼")

## User Input

geography = st.selectbox("Geography" , onehot_encoder_geo.categories_[0])

gender = st.selectbox("Gender" , label_encoder_gender.classes_)

age = st.slider("Age" , 18 , 92)

balance = st.number_input('Balance', min_value=0.0, step=100.0)

credit_score = st.number_input('Credit Score' , min_value=0)

estimated_salary = st.number_input('Estimated Salary', min_value=0.0 , step=500.0)

tenure = st.slider('Tenure', 0,10)

num_of_products = st.slider("Number of Products" , 1 , 4)

has_cr_card = st.selectbox("Has Credit Card" , [0,1])

is_active_member = st.selectbox("Is Active Member" , [0,1])


# Prepare the input data for prediction

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    "Tenure" : [tenure],
    'Balance' : [balance],
    "NumOfProducts" : [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' :[estimated_salary], 
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded , columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Combine one-hot encoded columns with the input data

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis=1)

# Scale the input data using the scaler
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0]

st.subheader("Prediction")

st.write(f"The probability of customer churn is: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.write("The Customer is likely to leave the bank")
else:
    st.write("The Customer is not likely to leave the bank")