import streamlit as st
import requests
import json

# Application Configuration
st.title("Air Quality Prediction Interface")
st.write("Enter the necessary data to predict air quality using the deployed model.")

# Define the endpoint URL and API key
ENDPOINT_URL = "https://air-quality-endpoint-mrw2.westeurope.inference.ml.azure.com/score"
API_KEY = "DcYh1EXGaVYZS1FNytWdrfYncEl4oMgo"

if not API_KEY:
    st.error("API key is missing. Please define it in the script.")
    st.stop()

# User Inputs
st.header("User Inputs")

temperature = st.number_input(
    "Temperature (°C)", min_value=-30.0, max_value=50.0, step=0.1, format="%.1f"
)
humidity = st.number_input(
    "Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"
)
pm25 = st.number_input(
    "PM2.5 Concentration (µg/m³)", min_value=0.0, step=0.1, format="%.1f"
)
pm10 = st.number_input(
    "PM10 Concentration (µg/m³)", min_value=0.0, step=0.1, format="%.1f"
)
no2 = st.number_input(
    "NO₂ Concentration (ppb)", min_value=0.0, step=0.1, format="%.1f"
)
so2 = st.number_input(
    "SO₂ Concentration (ppb)", min_value=0.0, step=0.1, format="%.1f"
)
co = st.number_input(
    "CO Concentration (ppm)", min_value=0.0, step=0.01, format="%.2f"
)
proximity = st.number_input(
    "Proximity to Industrial Areas (km)", min_value=0.0, step=0.1, format="%.1f"
)
density = st.number_input(
    "Population Density (people/km²)", min_value=0, step=1
)

# Function to prepare input data for the API
def prepare_input_data(*args):
    """Prepare the input data in the required format for the API."""
    return {"data": [list(args)]}

# Function to send a POST request to the API
def get_air_quality_prediction(input_data, endpoint, api_key):
    """
    Send a POST request to the air quality prediction API.

    Parameters:
        input_data (dict): The data to send in the request.
        endpoint (str): The API endpoint URL.
        api_key (str): The API authentication key.

    Returns:
        dict: The JSON response from the API if successful.
        None: If the request fails.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(input_data))
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err} - {response.text}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Request error occurred: {req_err}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    return None

# Button to send the request
if st.button("Predict"):
    # Prepare the input data for the API
    input_data = prepare_input_data(
        temperature, humidity, pm25, pm10, no2, so2, co, proximity, density
    )

    # Get the prediction from the API
    prediction_response = get_air_quality_prediction(input_data, ENDPOINT_URL, API_KEY)

    # Display the prediction result
    if prediction_response:
        # Extract predictions from the response
        predictions = prediction_response.get("predictions", [])
        
        if predictions:
            # Parse and display each prediction without a prefix
            st.success("Air Quality Predictions:")
            for pred in predictions:
                st.write(pred)  # Directly display the string
        else:
            st.error("No predictions found in the API response.")
    else:
        st.error("Failed to get a response from the API.")

