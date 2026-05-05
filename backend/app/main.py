import math
import os
import pickle
import faiss
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from .feature_store import FeatureStore

app = FastAPI()

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "..", "data", "index")
MODEL_DIR = os.path.join(BASE_DIR, "..", "data", "models")

# ── Load all assets once at startup ───────────────────────────────────────────
model          = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs={"use_safetensors": False})
index          = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.bin"))
feature_matrix = np.load(os.path.join(INDEX_DIR, "feature_matrix.npy"))   # (24010, 23)
ranker         = lgb.Booster(model_file=os.path.join(MODEL_DIR, "ranker.lgb"))
genre_matrix   = feature_matrix[:, :19]
feature_store  = FeatureStore()

with open(os.path.join(INDEX_DIR, "movie_lookup.pkl"), "rb") as f:
    movie_lookup = pickle.load(f)


def _clean(movie: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in movie.items()}


def _genre_affinity(candidate_indices: np.ndarray, user_id: int | None) -> np.ndarray:
    """
    Returns a (19,) genre affinity vector for scoring.

    With user_id  → fetch the user's precomputed genre preference from Redis.
    Without       → fall back to the mean genre vector of the top-20 FAISS hits,
                    which approximates the genre of the query itself.
    """
    if user_id is not None and feature_store.available:
        user_feats = feature_store.get_user_features(user_id)
        if user_feats is not None:
            return np.array(user_feats["genre_affinity"], dtype=np.float32)

    # Fallback: infer affinity from top-20 retrieved candidates
    return genre_matrix[candidate_indices[:20]].mean(axis=0)


@app.get("/")
def index_route():
    return {"message": "Hello World"}


@app.get("/search")
def get_movies(q: str, k: int = 10, user_id: int | None = None):
    # ── Stage 1: FAISS retrieval — fetch 100 candidates ───────────────────────
    query_vector = model.encode([q], normalize_embeddings=True).astype(np.float32)
    _, candidate_indices = index.search(query_vector, 100)
    candidate_indices = candidate_indices[0]                         # (100,)

    # ── Genre affinity: from Redis user profile or FAISS fallback ─────────────
    affinity = _genre_affinity(candidate_indices, user_id)           # (19,)

    # ── Build feature matrix for all 100 candidates ───────────────────────────
    X_items       = feature_matrix[candidate_indices]                # (100, 23)
    genre_overlap = (genre_matrix[candidate_indices] * affinity).sum(axis=1, keepdims=True)
    X             = np.hstack([X_items, genre_overlap]).astype(np.float32)  # (100, 24)

    # ── Stage 2: LightGBM re-ranking ──────────────────────────────────────────
    ranker_scores = ranker.predict(X)                                # (100,)
    top_positions = np.argsort(ranker_scores)[::-1][:k]

    results = []
    for pos in top_positions:
        movie = _clean(movie_lookup[candidate_indices[pos]])
        movie["score"] = round(float(ranker_scores[pos]), 4)
        results.append(movie)

    return results


@app.get("/user/{user_id}/features")
def get_user_features(user_id: int):
    """Inspect the feature store profile for a given user."""
    if not feature_store.available:
        return {"error": "Feature store unavailable"}
    feats = feature_store.get_user_features(user_id)
    if feats is None:
        return {"error": f"No profile found for user {user_id}"}
    return feats


@app.get("/feature-store/stats")
def feature_store_stats():
    return feature_store.stats()
