from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Charger le modèle entraîné
model = joblib.load("model.joblib")

@app.route('/')
def home():
    return "UR3 Cobot Predictive API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    try:
        # Vérification de la forme des données
        input_sequence = np.array(data['sequence'])  # doit être une liste de listes (10x22)

        if input_sequence.shape != (10, 22):
            return jsonify({"error": "La séquence doit avoir une forme (10, 22)"}), 400

        # Aplatir la séquence (si le modèle attend 1D)
        input_flattened = input_sequence.flatten().reshape(1, -1)

        prediction = model.predict(input_flattened)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
