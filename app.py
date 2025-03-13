from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("fish_market_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Load HTML page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract values from request
        species = float(data["species"])
        length1 = float(data["length1"])
        length2 = float(data["length2"])
        length3 = float(data["length3"])
        height = float(data["height"])
        width = float(data["width"])

        # Make prediction
        features = np.array([[species, length1, length2, length3, height, width]])
        predicted_weight = model.predict(features)[0]

        return jsonify({"prediction": round(predicted_weight, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=3001)
