import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Cargar datos
df = pd.read_csv("data/python_qa_dataset.csv")

# Vectorizar preguntas
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["pregunta"])
y = df["respuesta"]

# Guardar vectorizador y datos
joblib.dump(vectorizer, "models/vectorizer.pkl")
df.to_pickle("data/processed_data.pkl")
