import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# API configuration
WEATHER_API_KEY = "4d79f2da5973a1e175cfab477f714084"
SOILGRID_API_URL = "https://rest.isric.org/soilgrids/v2.0/query"

# Load ML model and related files
model = joblib.load("crop_recommendation_model.pkl")
encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")
scaler = joblib.load("scaler.pkl")

# Function to fetch coordinates from OpenWeatherMap
def get_coordinates(city):
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}"
    response = requests.get(weather_url)
    data = response.json()
    if "coord" in data:
        return data["coord"]["lat"], data["coord"]["lon"]
    return None, None

# Function to fetch weather data
def get_weather_data(lat, lon):
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(weather_url)
    data = response.json()
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    rainfall = data.get("rain", {}).get("1h", 50)
    return temperature, humidity, rainfall

# Function to fetch soil data from the updated SoilGrids API
def get_soil_data(lat, lon):
    soil_url = f"{SOILGRID_API_URL}?lon={lon}&lat={lat}"
    response = requests.get(soil_url)
    if response.status_code == 200:
        soil_json = response.json()
        properties = soil_json.get("properties", {}).get("soilproperties", {})
        nitrogen = properties.get("nitrogen", {}).get("value", [None])[0]
        phosphorus = properties.get("phosphorus", {}).get("value", [None])[0]
        potassium = properties.get("potassium", {}).get("value", [None])[0]
        ph = properties.get("phh2o", {}).get("value", [None])[0]
        return nitrogen, phosphorus, potassium, ph
    else:
        return None, None, None, None

# Function to predict top 3 crops from new data
def predict_crop(data):
    data_df = pd.DataFrame([data], columns=feature_names)
    data_scaled = scaler.transform(data_df)
    probabilities = model.predict_proba(data_scaled)[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_crops = encoder.inverse_transform(top_3_indices)
    return list(top_3_crops)

@app.route('/recommend', methods=['POST'])
def recommend_crop():
    try:
        data = request.json
        city = data.get('city')
        if not city:
            return jsonify({"error": "City is required"}), 400

        lat, lon = get_coordinates(city)
        if lat is None or lon is None:
            return jsonify({"error": "Invalid city name or unable to retrieve coordinates"}), 400

        temperature, humidity, rainfall = get_weather_data(lat, lon)
        nitrogen, phosphorus, potassium, ph = get_soil_data(lat, lon)
        if nitrogen is None or phosphorus is None or potassium is None or ph is None:
            nitrogen, phosphorus, potassium = 50, 30, 20
            ph = 6.5

        input_data = {
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }

        recommended_crops = predict_crop(input_data)

        return jsonify({
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "pH": ph,
            "recommended_crops": recommended_crops
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

