import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Ford Price Predictor", page_icon="🚗", layout="wide")

# Custom CSS to make it look modern
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_assets():
    """Load and cache the model and preprocessing objects."""
    model = joblib.load('car_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, scaler, columns

def get_prediction(data, model, scaler, columns):
    """Processes input data and returns a prediction."""
    # 1. Create a zero-filled DataFrame with correct columns
    input_df = pd.DataFrame(0, index=[0], columns=columns)
    
    # 2. Map Numerical values
    for col in ['year', 'mileage', 'tax', 'mpg', 'engineSize']:
        input_df[col] = data[col]
    
    input_df['is_new_tax_system'] = 1 if data['year'] >= 2017 else 0
    
    # 3. Handle One-Hot Encoding
    for cat_col in [f"model_{data['model']}", f"transmission_{data['transmission']}", f"fuelType_{data['fuelType']}"]:
        if cat_col in input_df.columns:
            input_df[cat_col] = 1

    # 4. Scale specific features
    cols_to_scale = ['mileage', 'tax', 'mpg', 'year']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    
    return model.predict(input_df)[0]

# --- MAIN APP ---
def main():
    model, scaler, columns = load_assets()

    st.title("🚗 Ford Used Car Price Predictor")
    st.markdown("---")

    # Sidebar for inputs (Cleaner UI)
    st.sidebar.header("📋 Car Specifications")
    
    user_data = {
        'model': st.sidebar.selectbox("Car Model", ['Focus', 'Fiesta', 'EcoSport', 'Kuga', 'S-MAX', 'Mondeo', 'others', 'C-MAX', 'KA', 'Galaxy', 'Edge', 'Puma', 'Mustang']),
        'year': st.sidebar.slider("Registration Year", 2000, 2025, 2018),
        'transmission': st.sidebar.radio("Transmission", ['Manual', 'Semi-Auto', 'Automatic']),
        'fuelType': st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Hybrid', 'Electric']),
        'mileage': st.sidebar.number_input("Total Mileage", min_value=0, value=25000),
        'tax': st.sidebar.number_input("Annual Tax (£)", min_value=0, value=145),
        'mpg': st.sidebar.number_input("Miles Per Gallon (MPG)", min_value=0.0, value=55.0),
        'engineSize': st.sidebar.slider("Engine Size (L)", 0.0, 5.0, 1.2)
    }

    # Display Input Summary in main area
    st.subheader("Selected Specifications")
    df_display = pd.DataFrame([user_data])
    st.table(df_display)

    if st.button("Calculate Market Value"):
        with st.spinner('Analyzing market data...'):
            result = get_prediction(user_data, model, scaler, columns)
            
            # Show result with a nice metric
            st.balloons()
            st.markdown("### 🎯 Estimated Valuation")
            st.metric(label="Predicted Price", value=f"£{result:,.2f}")
            
            st.info("Note: This is an AI-generated estimate based on historical Ford sales data.")

if __name__ == "__main__":
    main()





# First command on terminal to run the app: cd "C:\Users\premb\All AIML\Car Price Prediction"
# Second command: streamlit run app.py