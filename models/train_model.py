import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib

# Cargar datos procesados
df = pd.read_pickle("data/processed_data.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
X = vectorizer.transform(df["pregunta"])

# Entrenar modelo de búsqueda de vecinos más cercanos
model = NearestNeighbors(n_neighbors=1, algorithm="brute")
model.fit(X)

# Guardar modelo
joblib.dump(model, "models/qa_model.pkl")
