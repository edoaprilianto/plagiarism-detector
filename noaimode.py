import os
import string
import nltk
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

# FastAPI Setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi preprocessing teks
def preprocess_text(teks):
    teks = teks.lower()
    teks = teks.translate(str.maketrans("", "", string.punctuation))
    kata_kata = word_tokenize(teks)
    stop_words = set(stopwords.words("indonesian"))
    kata_kata = [kata for kata in kata_kata if kata not in stop_words]
    kata_kata = [stemmer.stem(kata) for kata in kata_kata]
    return " ".join(kata_kata)

# Fungsi cek kemiripan
def cek_kemiripan(teks1, teks2):
    teks1 = preprocess_text(teks1)
    teks2 = preprocess_text(teks2)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform([teks1, teks2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity * 100)

# Halaman utama
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Endpoint untuk memeriksa kemiripan teks
@app.post("/cek_kemiripan/")
async def process_text(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    teks1 = (await file1.read()).decode("utf-8")
    teks2 = (await file2.read()).decode("utf-8")
    skor = cek_kemiripan(teks1, teks2)
    return templates.TemplateResponse("index.html", {"request": request, "result": skor})