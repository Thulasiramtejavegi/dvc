from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("trained_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert flight type to numeric
        flight_type_map = {"economy": 0, "business": 1, "first_class": 2}
        flight_type_numeric = flight_type_map.get(data["flightType"].lower(), -1)

        if flight_type_numeric == -1:
            return jsonify({"error": "Invalid flight type"}), 400

        # Ensure all 16 features are provided
        features = [
            data["userCode"], 
            data["time"], 
            data["distance"], 
            flight_type_numeric
        ]

        # Add dummy values for the missing 12 features
        missing_features = [0] * 12  # Replace with real default values if needed
        features.extend(missing_features)

        features = np.array(features).reshape(1, -1)  # Convert to 2D array

        predicted_price = model.predict(features)[0]
        return jsonify({"predicted_price": predicted_price})

    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True)
