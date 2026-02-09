from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the model once at startup
MODEL_PATH = os.path.join("artifacts", "model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_pipeline.py first.")

pipeline = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get JSON data
        data = request.get_json()
        
        # Validate input (basic check)
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        # 2. Convert JSON to DataFrame
        # We expect data to be a dict like: 
        # {"carat": 1.52, "cut": "Premium", "color": "F", ...}
        # If sending multiple records, data should be a list of dicts.
        
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Input must be a JSON object or list of objects"}), 400

        # 3. Predict
        prediction = pipeline.predict(df)
        
        # 4. Return JSON response
        # Convert numpy array to list for JSON serialization
        return jsonify({
            "status": "success",
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)