import pandas as pd
import numpy as np
import joblib
import requests
import json

# Charger le dataset
df = pd.read_excel("dataset_02052023.xlsx")

# Nettoyage
df.columns = df.columns.str.strip()
df_cleaned = df.drop(columns=['Timestamp','Robot_ProtectiveStop'])

# Sélection de la séquence (10 lignes consécutives)
sequence = df_cleaned.iloc[100:110].copy()

# Prétraitement
#scaler = joblib.load("scaler.joblib")
#imputer = joblib.load("imputer.joblib")

#sequence_scaled = scaler.transform(sequence)

# Préparer le payload
payload = {
    "sequence": sequence.to_numpy().tolist()
}

# Appel POST à l'API Flask
response = requests.post("http://127.0.0.1:5000/predict", json=payload)

print("Prediction:", response.json())
print("Shape de la séquence envoyée:", sequence.shape)
