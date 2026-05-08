import hashlib
import json as _json
import math
import os
import pathlib
import pickle
import random
import threading
from datetime import datetime, timezone

import faiss
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from fastapi.middleware.cors import CORSMiddleware
try:
    from .feature_store import FeatureStore
except ImportError:
    from feature_store import FeatureStore  # when invoked without package context

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "..", "data", "index")
MODEL_DIR = os.path.join(BASE_DIR, "..", "data", "models")

# A/B logging setup 
LOG_DIR  = pathlib.Path(BASE_DIR) / ".." / "data" / "ab_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "events.jsonl"
_log_lock = threading.Lock()

# Load all assets once at startup 
model          = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs={"use_safetensors": False})
index          = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.bin"))
feature_matrix = np.load(os.path.join(INDEX_DIR, "feature_matrix.npy"))   # (24010, 23)
ranker         = lgb.Booster(model_file=os.path.join(MODEL_DIR, "ranker.lgb"))
genre_matrix   = feature_matrix[:, :19]
feature_store  = FeatureStore()

with open(os.path.join(INDEX_DIR, "movie_lookup.pkl"), "rb") as f:
    movie_lookup = pickle.load(f)

feature_movie_ids    = np.load(os.path.join(INDEX_DIR, "feature_movie_ids.npy"))
movie_id_to_feat_idx = {int(feature_movie_ids[i]): i for i in range(len(feature_movie_ids))}
movie_decade_map: dict[int, int] = {}

for entry in movie_lookup:
    yr = entry.get("year")
    if yr and not (isinstance(yr, float) and math.isnan(yr)):
        movie_decade_map[entry["movieId"]] = (int(yr) // 10) * 10


def _clean(movie: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in movie.items()}


def _user_features(candidate_indices: np.ndarray, user_id: int | None) -> dict:
    fallback_affinity = genre_matrix[candidate_indices[:20]].mean(axis=0)

    if user_id is not None and feature_store.available:
        feats = feature_store.get_user_features(user_id)
        if feats is not None:
            return {
                "genre_affinity"   : np.array(feats["genre_affinity"], dtype=np.float32),
                "avg_rating_norm"  : (feats["avg_rating_given"] - 0.5) / 4.5,
                "watch_count_norm" : min(np.log1p(feats["watch_count"]) / np.log1p(10_000), 1.0),
                "favorite_decade"  : feats["favorite_decade"],
            }

    return {
        "genre_affinity"  : fallback_affinity,
        "avg_rating_norm" : 0.5,
        "watch_count_norm": 0.5,
        "favorite_decade" : 0,
    }


# A/B helpers 

def _assign_variant(user_id: int | None) -> str:

    if user_id is not None:
        digest = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        return "baseline" if digest % 2 == 0 else "ranker"
    return random.choice(["baseline", "ranker"])


def _log_event(event: dict) -> None:
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(_json.dumps(event) + "\n")


@app.get("/")
def index_route():
    return {"message": "Hello World"}


@app.get("/search")
def get_movies(q: str, k: int = 10, user_id: int | None = None):
    # FAISS retrieval (100 candidates) 
    query_vector = model.encode([q], normalize_embeddings=True).astype(np.float32)
    faiss_scores, candidate_indices = index.search(query_vector, 100)
    faiss_scores      = faiss_scores[0]       # (100,)  cosine similarities
    candidate_indices = candidate_indices[0]  # (100,)

    # A/B variant assignment 
    variant = _assign_variant(user_id)

    if variant == "ranker":
        # Full pipeline: LightGBM re-ranking 
        uf = _user_features(candidate_indices, user_id)

        X_items = feature_matrix[candidate_indices]                    # (100, 23)

        genre_overlap = (genre_matrix[candidate_indices] * uf["genre_affinity"]).sum(axis=1)

        decade_match = np.array([
            1.0 if movie_decade_map.get(movie_lookup[i]["movieId"], -1) == uf["favorite_decade"]
            else 0.0
            for i in candidate_indices
        ], dtype=np.float32)

        user_cols = np.column_stack([
            genre_overlap,
            np.full(100, uf["avg_rating_norm"],  dtype=np.float32),
            np.full(100, uf["watch_count_norm"], dtype=np.float32),
            decade_match,
        ])                                                             # (100, 4)

        X = np.hstack([X_items, user_cols]).astype(np.float32)        # (100, 27)

        scores        = ranker.predict(X)
        top_positions = np.argsort(scores)[::-1][:k]
    else:
        # FAISS cosine similarity order only 
        top_positions = np.arange(min(k, len(candidate_indices)))
        scores        = faiss_scores

    results = []
    for pos in top_positions:
        movie = _clean(movie_lookup[candidate_indices[pos]])
        movie["score"] = round(float(scores[pos]), 4)
        results.append(movie)

    _log_event({
        "ts"           : datetime.now(timezone.utc).isoformat(),
        "user_id"      : user_id,
        "query"        : q,
        "variant"      : variant,
        "top_movie_ids": [r["movieId"] for r in results],
    })

    return {"variant": variant, "results": results}


class FeedbackBody(BaseModel):
    movie_id: int
    action: str   # "like" | "dislike"


@app.post("/user/{user_id}/feedback")
def submit_feedback(user_id: int, body: FeedbackBody):
    if body.action not in ("like", "dislike"):
        raise HTTPException(status_code=400, detail="action must be 'like' or 'dislike'")
    if not feature_store.available:
        raise HTTPException(status_code=503, detail="Feature store unavailable")

    feat_idx = movie_id_to_feat_idx.get(body.movie_id)
    if feat_idx is None:
        raise HTTPException(status_code=404, detail=f"Movie {body.movie_id} not in index")

    genre_vector = genre_matrix[feat_idx].tolist()
    movie_decade = movie_decade_map.get(body.movie_id)

    updated = feature_store.update_from_feedback(user_id, genre_vector, body.action, movie_decade)
    return {"ok": True, "top_genres": updated["top_genres"], "watch_count": updated["watch_count"]}


@app.get("/user/{user_id}/features")
def get_user_features(user_id: int):
    if not feature_store.available:
        return {"error": "Feature store unavailable - Redis not connected"}
    feats = feature_store.get_user_features(user_id)
    if feats is None:
        return {"error": f"No profile found for user {user_id}. Run populate_feature_store.py first."}
    return feats


@app.get("/feature-store/stats")
def feature_store_stats():
    return feature_store.stats()


@app.get("/ab/summary")
def ab_summary():
    if not LOG_FILE.exists():
        return {"total": 0, "by_variant": {}, "top_queries": []}

    counts: dict[str, int] = {}
    query_counts: dict[str, int] = {}
    total = 0

    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            v = ev.get("variant", "unknown")
            counts[v] = counts.get(v, 0) + 1
            q = ev.get("query", "")
            if q:
                query_counts[q] = query_counts.get(q, 0) + 1
            total += 1

    top_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total"      : total,
        "by_variant" : counts,
        "top_queries": [{"query": q, "count": c} for q, c in top_queries],
    }
