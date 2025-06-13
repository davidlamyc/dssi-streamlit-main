import streamlit as st
import pickle
import numpy as np

st.title("Resalse Price Prediction App")

# Input fields
bedrooms_input = st.number_input("Enter your bedrooms:", value=0)
floor_area_sqm_input = st.number_input("Enter your floor area sqm:", value=80)
lease_commence_date_input = st.number_input("Enter your lease commence date:", value=2000)
remaining_years_input = st.number_input("Enter your remaining years:", value=75)
storey_input = st.number_input("Enter your storey:", value=10)


filename = 'linear_regression_model_house.pkl'
# pickle.dump(model, open(filename, 'wb'))

# Load the model
loaded_model = pickle.load(open(filename, 'rb'))

# Use the loaded model for prediction
new_data = np.array([[bedrooms_input,floor_area_sqm_input,lease_commence_date_input,remaining_years_input,storey_input]])
prediction = loaded_model.predict(new_data)[0]
print(prediction)

# Display the inputs
# st.write(F"### You entered height of: {height_input}")
st.write(f"**Predicted Resale Price:** {prediction}")