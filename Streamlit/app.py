
import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# Load trained model
with open("car_price_xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the saved binary encoder
encoder = joblib.load("binary_encoder.joblib")

# Load the column mappings for one-hot encoding
city_columns = joblib.load('city_columns_mapping.joblib')
fuel_type_columns = joblib.load('fuel_type_columns_mapping.joblib')
body_type_columns = joblib.load('body_type_columns_mapping.joblib')
transmission_columns = joblib.load('transmission_columns_mapping.joblib')

# Load the minmax scalar
minmax_scaler = joblib.load("scaler.joblib")

# Streamlit UI with background and custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://s3.easternpeak.com/wp-content/uploads/2023/03/Top-Car-App-Ideas-for-the-Automotive-Industry1.jpg');
        background-size: cover;
        background-position: center;
    }
    .stTitle {
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: white;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
    }
    .stSelectbox, .stNumberInput {
        margin: 10px 0;
        font-size: 16px;
        color: white;
    }
    .stButton>button {
        background-color: #FF5733;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #C70039;
    }
    /* Styling for all input labels (headings) */
    label {
        color: white !important;
        font-size: 18px;
        font-weight: bold;
    }
    /* Custom styling for st.success */
    .st-alert-success {
        background-color: white;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# Header with title using HTML for custom styling
st.markdown("<h1 class='stTitle' style='color: white;'>Car Price Prediction</h1>", unsafe_allow_html=True)

# User input fields with better style and spacing
km = st.number_input("Kilometers Driven", min_value=0, max_value=155000, value=0, step=500)
ownerNo = st.number_input("Number of Owners", min_value=0, max_value=5, value=1, step=1)
manufacturer = st.selectbox("Manufacturer", ['Maruti', 'Ford', 'Tata', 'Hyundai', 'Jeep', 'Datsun', 'Honda', 'Mahindra', 'Mercedes-Benz', 'BMW', 'Renault', 'Audi', 'Toyota', 'Mini', 'Kia', 'Skoda', 'Volkswagen', 'Volvo', 'MG', 'Nissan', 'Fiat', 'Mahindra Ssangyong', 'Mitsubishi', 'Jaguar', 'Land Rover', 'Chevrolet', 'Citroen', 'Opel', 'Mahindra Renault', 'Isuzu', 'Lexus', 'Porsche', 'Hindustan Motors'])
engine_displacement = st.number_input("Engine Displacement (cc)", min_value=500, max_value=5000, value=500, step=100)
mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=35.5, value=5.0, step=1.0)
seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8, 9, 10])
no_of_cylinders = st.selectbox("Number of Cylinders", [2, 3, 4, 5, 6, 7, 8])
no_of_doors = st.selectbox("Number of Doors", [2, 3, 4, 5, 6])

city = st.selectbox("City", ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata'])
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Lpg', 'Cng', 'Electric'])
body_type = st.selectbox("Body Type", ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans', 'Pickup Trucks', 'Convertibles', 'Hybrids', 'Wagon'])
transmission = st.selectbox("Transmission", ['Automatic', 'Manual'])

# Function to apply one-hot encoding to input data during prediction
def apply_one_hot_encoding(input_data):
    input_data = pd.get_dummies(input_data, columns=['City', 'Fuel Type', 'Body Type', 'Transmission'], dtype=int)
    expected_columns = city_columns + fuel_type_columns + body_type_columns + transmission_columns
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)
    return input_data

# Convert input into a DataFrame before encoding
manufacturer_input = pd.DataFrame({'Manufacturer': [manufacturer]})

# Apply the SAME encoder used during training
manufacturer_encoded = encoder.transform(manufacturer_input)

# Ensure the encoded data has the expected columns
expected_binary_columns = ['Manufacturer_0', 'Manufacturer_1', 'Manufacturer_2', 'Manufacturer_3', 'Manufacturer_4', 'Manufacturer_5']
manufacturer_encoded = manufacturer_encoded.reindex(columns=expected_binary_columns, fill_value=0)

# Convert to list for model input
manufacturer_encoded = manufacturer_encoded.values.flatten().tolist()

# Prepare categorical input for encoding
categorical_data = pd.DataFrame({'City': [city], 'Fuel Type': [fuel_type], 'Body Type': [body_type], 'Transmission': [transmission]})

# Apply one-hot encoding to categorical data
one_hot_encoded = apply_one_hot_encoding(categorical_data).values.flatten().tolist()

# Normalize numerical features using MinMaxScaler
numerical_data = np.array([[km, engine_displacement, mileage]])
normalized_numerical = minmax_scaler.transform(numerical_data).flatten().tolist()

# Combine all features
input_features = normalized_numerical + [ownerNo] + manufacturer_encoded + [seats, no_of_cylinders, no_of_doors] + one_hot_encoded

print("Feature vector length:", len(input_features))  # Debugging step

# Predict button
if st.button("Predict Car Price"):
    input_array = np.array(input_features)
    print("Input array shape:", input_array.shape)  # Debugging step
    input_array = input_array.reshape(1, -1)
    predicted_price = model.predict(input_array)[0]
    st.success(f"Estimated Car Price: ₹{predicted_price:,.2f}")
