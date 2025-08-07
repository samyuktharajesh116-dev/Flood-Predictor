import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
with open('flood_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(r'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


st.title('Flood Prediction App')

# User input
normal = st.number_input('Enter Normal value')
average = st.number_input('Enter Average value')
deviation = st.number_input('Enter Deviation value')

if st.button('Predict'):
    input_data = np.array([[normal, average, deviation]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    if prediction[0] == 1:
        st.error('⚠️ Flood likely!')
    else:
        st.success('✅ No flood predicted.')
