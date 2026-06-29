from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import requests
from openai import OpenAI

app = Flask(__name__)
CORS(app)   



client = OpenAI(
    api_key="",
    base_url="https://api.groq.com/openai/v1"
)



crop_model   = pickle.load(open('crop_model.pkl', 'rb'))
fert_model   = pickle.load(open('models/fertilizer_model.pkl', 'rb'))
soil_encoder = pickle.load(open('models/soil_encoder.pkl', 'rb'))
crop_encoder = pickle.load(open('models/crop_encoder.pkl', 'rb'))
fert_encoder = pickle.load(open('models/fertilizer_encoder.pkl', 'rb'))



def safe_encode(encoder, value):
    value = str(value).strip().capitalize()
    if value not in encoder.classes_:
        return encoder.transform([encoder.classes_[0]])[0]
    return encoder.transform([value])[0]



def get_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )
    resp = requests.get(url, timeout=8)
    resp.raise_for_status()
    return resp.json()['current_weather']



def get_coordinates(city):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    res = requests.get(url, timeout=8).json()
    if 'results' in res and res['results']:
        r = res['results'][0]
        return r['latitude'], r['longitude']
    return 18.6, 73.8   # default: Pune



def predict_fertilizer(temp, moisture, rainfall, ph, N, P, K, carbon, soil, crop):
    soil_enc = safe_encode(soil_encoder, soil)
    crop_enc = safe_encode(crop_encoder, crop)
    features = [[temp, moisture, rainfall, ph, N, P, K, carbon, soil_enc, crop_enc]]
    pred = fert_model.predict(features)[0]
    return fert_encoder.inverse_transform([pred])[0]



def generate_ai_content(crop, weather, N, P, K):
    prompt = f"""
You are an expert agronomist. Based on the data below, give structured advice.

Crop: {crop}
Temperature: {weather.get('temperature', 'N/A')}°C
Wind speed: {weather.get('windspeed', 'N/A')} km/h
Soil Nutrients — Nitrogen: {N}, Phosphorus: {P}, Potassium: {K}

Respond in EXACTLY this format (keep each section header on its own line):

Explanation:
<2-3 sentences explaining why this crop suits the conditions>

Process:
1. <step>
2. <step>
3. <step>
4. <step>
5. <step>

Tips:
- <tip>
- <tip>
- <tip>
- <tip>
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )
        return response.choices[0].message.content

    except Exception as e:
        print("Groq AI ERROR:", e)
        return f"""Explanation:
{crop.capitalize()} is well-suited for the current soil and climate conditions.
The nutrient levels of N={N}, P={P}, K={K} support healthy growth.

Process:
1. Prepare soil and adjust pH to optimal range
2. Apply recommended fertilizer before sowing
3. Sow seeds at correct depth and spacing
4. Set up irrigation schedule based on rainfall
5. Monitor and harvest at peak maturity

Tips:
- Rotate crops each season to maintain soil health
- Use organic compost to improve soil structure
- Monitor weather forecasts before fertilizer application
- Keep detailed field records for future planning
"""



def estimate_yield(N, P, K, temp, rainfall):
    return round((N + P + K) * 0.5 + rainfall * 0.3 + temp * 0.2, 2)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        N        = float(request.form['N'])
        P        = float(request.form['P'])
        K        = float(request.form['K'])
        temp     = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph       = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        soil     = request.form.get('soil', 'Loamy')
        city     = request.form.get('city', 'Pune')

        lat, lon = get_coordinates(city)
        weather  = get_weather(lat, lon)

        input_df = pd.DataFrame([{
            "N": N, "P": P, "K": K,
            "temperature": temp,
            "humidity":    humidity,
            "ph":          ph,
            "rainfall":    rainfall,
        }])
        crop = crop_model.predict(input_df)[0]

        fertilizer = predict_fertilizer(
            temp, 40, rainfall, ph, N, P, K, 1.0, soil, crop
        )

        yield_value = estimate_yield(N, P, K, temp, rainfall)

        ai_content = generate_ai_content(crop, weather, N, P, K)

        return jsonify({
            "crop":       crop,
            "fertilizer": fertilizer,
            "weather":    weather,      
            "yield":      yield_value,
            "ai":         ai_content,
            "N": N, "P": P, "K": K,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)