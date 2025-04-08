# Projet_I4.0_MLxDL.ECC

OUATTARA Amadou
Projet pour l'évaluation finale dans le cadre du cours sur le ML et DL dans l'industrie 4.0 à L'ECC.


# UR3 Cobot Predictive Maintenance API

Ce projet implémente une API de prédiction pour la maintenance prédictive d'un robot collaboratif UR3, permettant d'anticiper les arrêts de protection du robot basé sur des séquences de données opérationnelles.

## Description du projet

Le système analyse des séquences de données collectées pendant le fonctionnement du robot UR3 pour prédire si un arrêt de protection va se produire. Cette prédiction permet d'anticiper les problèmes et d'intervenir de manière proactive avant qu'un arrêt ne se produise.

## Technologies utilisées

- **Python 3.x**
- **TensorFlow** : Pour la création et l'entraînement du modèle LSTM
- **Flask** : Pour l'API REST
- **Pandas/NumPy** : Pour la manipulation des données
- **Scikit-learn** : Pour le prétraitement des données

## Architecture du projet

- **Modèle ML** : Un réseau LSTM (Long Short-Term Memory) qui analyse des séquences temporelles de données du robot
- **API REST** : Interface permettant de recevoir des séquences de données et de retourner des prédictions
- **Prétraitement** : Pipeline de transformation des données brutes en features utilisables par le modèle

## Installation

1. Cloner le répertoire

2. Créer et activer un environnement virtuel :
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Fichiers de configuration

Assurez-vous d'avoir les fichiers suivants dans votre répertoire de travail(ces fichiers sont deja dans ce repo) :
- `model.h5` ou `model.joblib` : Modèle entraîné
- `scaler.joblib` : Normalisation des données
- `imputer.joblib` : Gestion des valeurs manquantes

## Utilisation de l'API

### Démarrer le serveur

```bash
python app.py
```

Le serveur démarre par défaut sur http://localhost:5000/

### Routes disponibles

1. **GET /** - Page d'accueil, vérifie que l'API fonctionne
2. **POST /predict** - Envoyer une séquence pour prédiction
3. **GET /test** - Test avec un échantillon du dataset

### Exemple de requête pour /predict

```python
import requests
import json
import numpy as np

# Créer une séquence test (10 pas de temps, 22 features)
sequence = np.random.random((10, 22)).tolist()

# Envoyer la requête
response = requests.post('http://localhost:5000/predict', 
                         json={'sequence': sequence})

# Afficher le résultat
print(response.json())
```

## Dépannage

### Problèmes courants

1. **Erreur lors du chargement du modèle**

   Si vous rencontrez l'erreur `ModuleNotFoundError: No module named 'tensorflow.python'`, essayez ces solutions:
   - Réinstallez TensorFlow avec la commande `pip install tensorflow`
   - Sauvegardez le modèle au format natif Keras (.h5) plutôt qu'avec joblib

2. **Incompatibilité de dimensions**

   Si le modèle attend des données de forme différente:
   - Vérifiez la forme attendue avec `print(model.input_shape)`
   - Assurez-vous que votre séquence d'entrée correspond à cette forme

3. **Valeurs manquantes dans les données**

   L'imputer doit être appliqué exactement de la même manière que lors de l'entraînement.

## Développement

### Structure des fichiers

```
├── app.py                  # Application Flask
├── model.h5                # Modèle entraîné (format Keras)
├── scaler.joblib           # Scaler pour la normalisation
├── imputer.joblib          # Imputer pour valeurs manquantes
├── requirements.txt        # Dépendances du projet
└── README.md               # Documentation
```

### To-Do

- [ ] Ajouter une interface utilisateur web

## Contact

Pour toute question concernant ce projet, veuillez contacter amadou.ouattara@centrale.casablanca.ma