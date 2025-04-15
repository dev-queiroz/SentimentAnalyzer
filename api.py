from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import nltk
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Inicializar o FastAPI
app = FastAPI()

# Configurar pré-processamento
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = emoji.demojize(text)  # Converte emojis em texto
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove menções
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove pontuações e números
    text = text.lower()  # Converte para minúsculas
    text = text.split()  # Tokenização
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words and len(word) > 2]
    return ' '.join(text)

# Carregar o modelo e o vetorizador
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Definir o modelo de entrada
class TextInput(BaseModel):
    text: str

# Endpoint para previsão
@app.post("/predict/")
async def predict_sentiment(input: TextInput):
    text = preprocess_text(input.text)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return {"sentiment": "Positivo" if prediction[0] == 1 else "Negativo"}

# Instrução para rodar: uvicorn api:app --reload