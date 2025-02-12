import joblib
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class PythonQABot:
    def __init__(
        self,
        data_path="data/processed_data.pkl",
        vectorizer_path="models/vectorizer.pkl",
        model_path="models/qa_model.pkl",
    ):

        try:
            self.df = pd.read_pickle(data_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.model = joblib.load(model_path)
            self._preprocess_data()  # <-- Método modificado
        except FileNotFoundError as e:
            raise RuntimeError(f"Error loading model files: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Initialization error: {str(e)}")

    def _preprocess_data(self):
        """Preprocesamiento inicial del dataset"""
        self.df["keywords"] = self.df["pregunta"].apply(
            lambda x: set(
                re.sub(r"[^\w\s]", "", x.lower()).split()
            )  # <-- Paréntesis cerrado
        )

    def _preprocess_text(self, text):
        """Limpieza de texto para queries"""
        return set(re.sub(r"[^\w\s]", "", text.lower()).split())

    def _find_keyword_match(self, query):
        """Búsqueda por coincidencia de palabras clave"""
        query_keywords = self._preprocess_text(query)
        best_match = None
        max_score = 0

        for _, row in self.df.iterrows():
            score = len(query_keywords & row["keywords"])
            if score > max_score:
                max_score = score
                best_match = row["respuesta"]
        return best_match if max_score > 0 else None

    def ask(self, question):
        """
        Método principal para obtener respuestas
        Args:
            question (str): Pregunta del usuario
        Returns:
            str: Respuesta del bot
        """
        if not isinstance(question, str) or len(question.strip()) == 0:
            raise ValueError("Invalid question format")

        # Primera búsqueda por keywords
        keyword_answer = self._find_keyword_match(question)
        if keyword_answer:
            return keyword_answer

        # Búsqueda por similitud semántica
        try:
            question_vec = self.vectorizer.transform([question])
            _, indices = self.model.kneighbors(question_vec)
            return self.df.iloc[indices[0][0]]["respuesta"]
        except Exception as e:
            raise RuntimeError(f"Error processing question: {str(e)}")


# import re
# import joblib
# import pandas as pd
# import random
# from textblob import TextBlob
# from gensim.models import FastText
# from sklearn.neighbors import NearestNeighbors
# import string
# import numpy as np


# # --------------------------------------------------
# # Módulo: Corrección Ortográfica
# # --------------------------------------------------
# class SpellChecker:
#     def __init__(self):
#         self.common_typos = {
#             "fnciones": "funciones",
#             "pythom": "python",
#             "clasee": "clase",
#             "funsion": "funcion",
#         }

#     def correct(self, text):
#         # Corrección manual de errores comunes
#         for typo, correct in self.common_typos.items():
#             text = re.sub(r"\b" + typo + r"\b", correct, text, flags=re.IGNORECASE)

#         # Corrección automática general
#         return str(TextBlob(text).correct())


# # --------------------------------------------------
# # Módulo: Embeddings y Modelo
# # --------------------------------------------------
# class EmbeddingModel:
#     def __init__(self):
#         self.model = None
#         self.nn_model = NearestNeighbors(n_neighbors=3, metric="cosine")

#     def train(self, sentences):
#         # Entrenar modelo FastText con datos ruidosos
#         noisy_data = self._add_noise(sentences)
#         self.model = FastText(vector_size=100, window=5, min_count=1)
#         self.model.build_vocab(noisy_data)
#         self.model.train(noisy_data, total_examples=len(noisy_data), epochs=10)

#         # Entrenar modelo de búsqueda
#         embeddings = [
#             self.model.wv[sentence.split()].mean(axis=0) for sentence in sentences
#         ]
#         self.nn_model.fit(embeddings)

#     def _add_noise(self, sentences, noise_level=0.3):
#         # Generar variaciones con errores tipográficos
#         noisy = []
#         for sent in sentences:
#             if len(sent) > 5 and random.random() < noise_level:
#                 noisy.append(self._introduce_typo(sent))
#             noisy.append(sent)
#         return noisy

#     def _introduce_typo(self, word):
#         # Lógica para introducir errores aleatorios
#         typo_ops = [self._delete_char, self._swap_chars, self._add_char]
#         return random.choice(typo_ops)(word)

#     def _delete_char(self, word):
#         if len(word) < 3:
#             return word
#         pos = random.randint(0, len(word) - 1)
#         return word[:pos] + word[pos + 1 :]

#     def _swap_chars(self, word):
#         if len(word) < 3:
#             return word
#         chars = list(word)
#         pos = random.randint(0, len(chars) - 2)
#         chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
#         return "".join(chars)

#     def _add_char(self, word):
#         if len(word) < 3:
#             return word
#         pos = random.randint(0, len(word) - 1)
#         random_char = random.choice(string.ascii_lowercase)
#         return word[:pos] + random_char + word[pos:]


# # --------------------------------------------------
# # Módulo Principal del Bot
# # --------------------------------------------------
# class PythonQABot:
#     def __init__(self):
#         # Cargar datos
#         self.df = pd.read_pickle("data/processed_data.pkl")

#         # Inicializar módulos
#         self.spell_checker = SpellChecker()
#         self.embedder = EmbeddingModel()
#         self._train_models()

#         # Historial de contexto
#         self.context = []

#     def _train_models(self):
#         # Preprocesar preguntas
#         self.df["clean_questions"] = self.df["pregunta"].apply(self._preprocess)

#         # Entrenar embeddings
#         self.embedder.train(self.df["clean_questions"].tolist())

#     def _preprocess(self, text):
#         # Normalización avanzada
#         text = text.lower().strip()
#         text = re.sub(r"[^\w\s]", "", text)
#         text = self.spell_checker.correct(text)
#         return text

#     def _get_embedding(self, text):
#         return self.embedder.model.wv[text.split()].mean(axis=0)

#     def ask(self, question):
#         try:
#             # Paso 1: Corrección y normalización
#             cleaned_q = self._preprocess(question)

#             # Paso 2: Búsqueda semántica
#             emb = self._get_embedding(cleaned_q)
#             distances, indices = self.embedder.nn_model.kneighbors([emb])

#             # Paso 3: Gestión de contexto
#             self._update_context(cleaned_q, indices[0])

#             # Paso 4: Selección de respuesta
#             return self._select_best_answer(indices[0], distances[0])

#         except Exception as e:
#             return f"Error procesando la pregunta: {str(e)}"

#     def _update_context(self, question, indices):
#         # Mantener historial de las últimas 3 interacciones
#         self.context = (self.context + [(question, indices)])[-3:]

#     def _select_best_answer(self, indices, distances):
#         # Priorizar respuestas basadas en contexto
#         if len(self.context) > 1:
#             last_question, last_indices = self.context[-2]
#             overlapping = set(indices) & set(last_indices)
#             if overlapping:
#                 return self.df.iloc[overlapping.pop()]["respuesta"]

#         # Seleccionar la mejor coincidencia
#         min_dist_idx = np.argmin(distances)
#         return self.df.iloc[indices[min_dist_idx]]["respuesta"]


# # --------------------------------------------------
# # Módulo: Interfaz de Usuario
# # --------------------------------------------------
# class ChatInterface:
#     def __init__(self, bot):
#         self.bot = bot
#         self.history = []

#     def start_chat(self):
#         print("Bot de Python (escribe 'salir' para terminar)\n")
#         while True:
#             user_input = input("Tú: ").strip()
#             if user_input.lower() == "salir":
#                 break
#             response = self.bot.ask(user_input)
#             self._display_response(response)

#     def _display_response(self, response):
#         print(f"\n╭{'─' * 50}╮")
#         print(f"│ {'Bot':^48} │")
#         print(f"├{'─' * 50}┤")
#         for line in response.split("\n"):
#             print(f"│ {line.ljust(48)} │")
#         print(f"╰{'─' * 50}╯\n")


# # --------------------------------------------------
# # Ejecución Principal
# # --------------------------------------------------
# if __name__ == "__main__":
#     bot = PythonQABot()
#     chat = ChatInterface(bot)
#     chat.start_chat()
