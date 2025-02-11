# Python QA Bot 🤖

Bot de preguntas y respuestas en Python entrenado para resolver dudas sobre programación.

## 🚀 Características

- Preprocesamiento de datos automatizado
- Modelo de NLP entrenado con scikit-learn
- Interfaz de línea de comandos interactiva
- Gestión de dependencias con pip

## 📦 Instalación

### Requisitos

- Python 3.8+
- pip

### Pasos:

1. Clonar repositorio:

```bash
- git clone https://github.com/tuusuario/python_qa_bot.git
- cd python_qa_bot
```

2. Crear y activar entorno virtual (recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

3. Dependencias:

```bash
pandas==1.3.3
scikit-learn==1.0.2
joblib==1.1.0
textblob==0.15.3
gensim==4.3.2
numpy==1.21.6
```

4. Uso

```bash
# Preprocesar datos
python data/preprocess.py

# Entrenar modelo
python models/train_model.py

# Iniciar chatbot
python src/chatbot.py

---------------------------------------------------------------------
Ejemplo
---------------------------------------------------------------------

Usuario > ¿Qué es una lista en Python?
Bot > Una colección ordenada y mutable de elementos. Se define con corchetes: [1, 'a', True]
```

```bash
python_qa_bot/
├── data/                   # Datos crudos y procesados
│   ├── python_qa_dataset.csv
│   └── processed_data.pkl
├── models/                # Modelos entrenados
│   ├── qa_model.pkl
│   └── vectorizer.pkl
├── src/                   # Código fuente
│   ├── chatbot.py         # Interfaz principal
│   └── utils.py           # Funciones auxiliares
├── venv/                  # Entorno virtual
├── .gitignore
├── README.md
└── requirements.txt
```

📧 Contacto
[Joan Iribe] - [jm_im@outlook.com]

---

Este README incluye:

- Instrucciones claras de instalación y uso
- Estructura de archivos visual
- Dependencias específicas con versiones
- Flujo de trabajo completo desde preprocesamiento hasta ejecución
- Secciones estándar para contribuciones y licencia

¡Personaliza los datos de contacto y añade capturas de pantalla si lo necesitas! 🚀
