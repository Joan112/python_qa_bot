from flask import Flask, request, jsonify
from flask_cors import CORS
from src.chatbot import PythonQABot

app = Flask(__name__)
CORS(app)  # Configuración para desarrollo

# Configuración de rutas relativas
bot = PythonQABot(
    data_path="data/processed_data.pkl",
    vectorizer_path="models/vectorizer.pkl",
    model_path="models/qa_model.pkl",
)


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/api/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Empty question"}), 400

        response = bot.ask(question)
        return jsonify({"question": question, "response": response, "error": None})

    except Exception as e:
        return jsonify({"question": question, "response": None, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
