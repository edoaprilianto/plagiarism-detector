import torch
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Load Model SBERT (Bahasa Indonesia)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# FastAPI Setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Fungsi cek kemiripan dengan SBERT
def cek_kemiripan(teks1, teks2):
    emb1 = model.encode(teks1, convert_to_tensor=True)
    emb2 = model.encode(teks2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return round(similarity * 100, 2)  # Persentase akurasi

# Halaman utama
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Endpoint untuk cek kemiripan teks
@app.post("/cek_kemiripan/")
async def process_text(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    teks1 = (await file1.read()).decode("utf-8")
    teks2 = (await file2.read()).decode("utf-8")
    skor = cek_kemiripan(teks1, teks2)
    return templates.TemplateResponse("index.html", {"request": request, "result": skor})
