import math
import os
import pickle
import faiss
import numpy as np
import torch
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "..", "data", "index")

model = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs={"torch_dtype": torch.float32})
index = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.bin"))

with open(os.path.join(INDEX_DIR, "movie_lookup.pkl"), "rb") as f:
    movie_lookup = pickle.load(f)


@app.get("/")
def index_route():
    return {"message": "Hello World"}


@app.get("/search")
def get_movies(q: str, k: int = 100):
    query_vector = model.encode([q], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_vector, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        movie = {
            key: (None if isinstance(val, float) and math.isnan(val) else val)
            for key, val in movie_lookup[idx].items()
        }
        movie["score"] = round(float(score), 4)
        results.append(movie)

    return results
