import streamlit as st
import pickle
import numpy as np

st.title("Weight Prediction App")

# Input fields
height_input = st.number_input("Enter your height:", value=0.0)

filename = 'linear_regression_model.pkl'
# pickle.dump(model, open(filename, 'wb'))

# Load the model
loaded_model = pickle.load(open(filename, 'rb'))

# Use the loaded model for prediction
new_data = np.array([[height_input]])
prediction = loaded_model.predict(new_data)[0]
print(prediction)

# Display the inputs
st.write(F"### You entered height of: {height_input}")
st.write(f"**new data:** {prediction}")