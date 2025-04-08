from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import tensorflow.keras.models as tf
app = Flask(__name__)

# === Chargement des éléments ===
model = tf.load_model("model.h5")
imputer = joblib.load("imputer.joblib")
scaler = joblib.load("scaler.joblib")

# === Prétraitement d'une séquence depuis le dataset ===
def preprocess_sample_sequence(csv_path, start_index=100):
    try:
        df = pd.read_excel(csv_path)

        # Sélectionner 10 lignes consécutives
        sequence = df.iloc[start_index:start_index+10].copy()

        # 💡 Optionnel : supprimer des colonnes inutiles
        sequence.drop(columns=['timestamp', 'target'], inplace=True)

        # Imputation
        #sequence_imputed = imputer.transform(sequence)

        # Scaling
        #sequence_scaled = scaler.transform(sequence_imputed)

        # Reshape
        if sequence.shape != (10, 22):
            raise ValueError(f"Mauvaise forme : attendu (10, 22), obtenu {sequence.shape}")

        return sequence

    except Exception as e:
        raise e

# === API ===

@app.route('/')
def home():
    return "UR3 Cobot Predictive API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    try:
        input_sequence = np.array(data['sequence'])

        if input_sequence.shape != (10, 22):
            return jsonify({"error": "La séquence doit avoir une forme (10, 22)"}), 400

        input_flattened = input_sequence.flatten().reshape(1, -1)
        prediction = model.predict(input_flattened)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test_sample():
    try:
        sequence = preprocess_sample_sequence("dataset_02052023.xlsx", start_index=100)
        input_flattened = sequence.flatten().reshape(1, -1)
        prediction = model.predict(input_flattened)
        return jsonify({
            "test_sequence_start_index": 100,
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Lancement ===
if __name__ == '__main__':
    app.run(debug=True)
