import math
import os
import pickle
import faiss
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "..", "data", "index")
MODEL_DIR = os.path.join(BASE_DIR, "..", "data", "models")

# ── Load all assets once at startup ───────────────────────────────────────────
model          = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs={"use_safetensors": False})
index          = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.bin"))
feature_matrix = np.load(os.path.join(INDEX_DIR, "feature_matrix.npy"))   # (24010, 23)
ranker         = lgb.Booster(model_file=os.path.join(MODEL_DIR, "ranker.lgb"))
genre_matrix   = feature_matrix[:, :19]   # genre multi-hot slice, reused every request

with open(os.path.join(INDEX_DIR, "movie_lookup.pkl"), "rb") as f:
    movie_lookup = pickle.load(f)


def _clean(movie: dict) -> dict:
    """Replace NaN float values with None so FastAPI can serialise them."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in movie.items()}


@app.get("/")
def index_route():
    return {"message": "Hello World"}


@app.get("/search")
def get_movies(q: str, k: int = 10):
    # ── Stage 1: FAISS retrieval — fetch 100 candidates ───────────────────────
    query_vector = model.encode([q], normalize_embeddings=True).astype(np.float32)
    _, candidate_indices = index.search(query_vector, 100)
    candidate_indices = candidate_indices[0]           # shape (100,)

    # ── Genre affinity: mean genre vector of top-20 FAISS hits ────────────────
    # Approximates "what genre does this query belong to?" without a user profile.
    query_genre_affinity = genre_matrix[candidate_indices[:20]].mean(axis=0)  # (19,)

    # ── Build feature matrix for all 100 candidates ───────────────────────────
    X_items      = feature_matrix[candidate_indices]                          # (100, 23)
    genre_overlap = (genre_matrix[candidate_indices] * query_genre_affinity).sum(axis=1, keepdims=True)
    X            = np.hstack([X_items, genre_overlap]).astype(np.float32)    # (100, 24)

    # ── Stage 2: Ranker re-scores ─────────────────────────────────────────────
    ranker_scores = ranker.predict(X)                                         # (100,)

    # ── Return top-k by ranker score ──────────────────────────────────────────
    top_positions = np.argsort(ranker_scores)[::-1][:k]

    results = []
    for pos in top_positions:
        movie = _clean(movie_lookup[candidate_indices[pos]])
        movie["score"] = round(float(ranker_scores[pos]), 4)
        results.append(movie)

    return results
