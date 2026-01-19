from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
with open("model_xyz.p", "rb") as f:
    model = pickle.load(f)["model"]

THRESHOLD = 0.50  # confidence threshold

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    landmarks = data["landmarks"]  # 63 values (x,y,z)

    X = np.array(landmarks).reshape(1, -1)

    # Predict probabilities
    probs = model.predict_proba(X)[0]
    max_prob = float(np.max(probs))
    pred_class = model.classes_[np.argmax(probs)]

    if max_prob < THRESHOLD:
        return jsonify({
            "prediction": "Invalid Sign",
            "confidence": max_prob
        })

    return jsonify({
        "prediction": str(pred_class),
        "confidence": max_prob
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
