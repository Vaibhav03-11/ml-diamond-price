from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}, methods=["POST", "OPTIONS"])

# Load the model once at startup
MODEL_PATH = os.path.join("artifacts", "model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_pipeline.py first.")

pipeline = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Input must be a JSON object or list of objects"}), 400

        prediction = pipeline.predict(df)
        return jsonify({"status": "success", "prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
