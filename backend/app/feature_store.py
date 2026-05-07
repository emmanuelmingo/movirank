"""
CineRank - Redis Feature Store Client

Key schema:
  user:{userId}  → Hash
    genre_affinity   : JSON float[19]  — normalised genre preference vector
    top_genres       : JSON str[]      — top-3 preferred genres by name
    favorite_decade  : int             — e.g. 1990
    avg_rating_given : float           — mean rating the user gives
    watch_count      : int             — total movies rated

  item:{movieId}  → Hash
    genre_vector     : JSON float[19]  — genre multi-hot
    avg_rating_norm  : float
    rating_count_norm: float
    popularity_norm  : float
    recency_norm     : float
    feat_idx         : int             — row index in feature_matrix.npy

All keys are given a 7-day TTL so stale profiles expire automatically.
User features for unseen users return None; callers fall back to defaults.
"""

import json
import os
import numpy as np
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
USER_TTL   = 7 * 24 * 3600   # 7 days
ITEM_TTL   = 7 * 24 * 3600

GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


class FeatureStore:
    def __init__(self):
        self._client = None
        self._available = False
        try:
            self._client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT,
                decode_responses=True, socket_connect_timeout=2,
            )
            self._client.ping()
            self._available = True
            print(f"[FeatureStore] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            print(f"[FeatureStore] Redis unavailable ({e}). Running without feature store.")

    @property
    def available(self) -> bool:
        return self._available

    # ── User features ──────────────────────────────────────────────────────────

    def get_user_features(self, user_id: int) -> dict | None:
        if not self._available:
            return None
        raw = self._client.hgetall(f"user:{user_id}")
        if not raw:
            return None
        return {
            "genre_affinity"  : json.loads(raw["genre_affinity"]),
            "top_genres"      : json.loads(raw["top_genres"]),
            "favorite_decade" : int(raw["favorite_decade"]),
            "avg_rating_given": float(raw["avg_rating_given"]),
            "watch_count"     : int(raw["watch_count"]),
        }

    def set_user_features(self, user_id: int, features: dict) -> None:
        if not self._available:
            return
        key = f"user:{user_id}"
        self._client.hset(key, mapping={
            "genre_affinity"  : json.dumps(features["genre_affinity"]),
            "top_genres"      : json.dumps(features["top_genres"]),
            "favorite_decade" : features["favorite_decade"],
            "avg_rating_given": features["avg_rating_given"],
            "watch_count"     : features["watch_count"],
        })
        self._client.expire(key, USER_TTL)

    # ── Item features ──────────────────────────────────────────────────────────

    def get_item_features(self, movie_id: int) -> dict | None:
        if not self._available:
            return None
        raw = self._client.hgetall(f"item:{movie_id}")
        if not raw:
            return None
        return {
            "genre_vector"     : json.loads(raw["genre_vector"]),
            "avg_rating_norm"  : float(raw["avg_rating_norm"]),
            "rating_count_norm": float(raw["rating_count_norm"]),
            "popularity_norm"  : float(raw["popularity_norm"]),
            "recency_norm"     : float(raw["recency_norm"]),
            "feat_idx"         : int(raw["feat_idx"]),
        }

    def set_item_features(self, movie_id: int, features: dict) -> None:
        if not self._available:
            return
        key = f"item:{movie_id}"
        self._client.hset(key, mapping={
            "genre_vector"     : json.dumps(features["genre_vector"]),
            "avg_rating_norm"  : features["avg_rating_norm"],
            "rating_count_norm": features["rating_count_norm"],
            "popularity_norm"  : features["popularity_norm"],
            "recency_norm"     : features["recency_norm"],
            "feat_idx"         : features["feat_idx"],
        })
        self._client.expire(key, ITEM_TTL)

    # ── Batch helpers ──────────────────────────────────────────────────────────

    def get_item_features_batch(self, movie_ids: list[int]) -> dict[int, dict]:
        """Returns {movieId: features} for all ids found in Redis."""
        if not self._available or not movie_ids:
            return {}
        pipe = self._client.pipeline(transaction=False)
        for mid in movie_ids:
            pipe.hgetall(f"item:{mid}")
        results = {}
        for mid, raw in zip(movie_ids, pipe.execute()):
            if raw:
                results[mid] = {
                    "genre_vector"     : json.loads(raw["genre_vector"]),
                    "avg_rating_norm"  : float(raw["avg_rating_norm"]),
                    "rating_count_norm": float(raw["rating_count_norm"]),
                    "popularity_norm"  : float(raw["popularity_norm"]),
                    "recency_norm"     : float(raw["recency_norm"]),
                    "feat_idx"         : int(raw["feat_idx"]),
                }
        return results

    def update_from_feedback(self, user_id: int, genre_vector: list[float],
                              action: str, movie_decade: int | None = None) -> dict | None:
        """
        Incrementally update a user's genre affinity from a like/dislike signal.

        Like    → nudge affinity toward the movie's genres  (+ALPHA per genre present)
        Dislike → nudge affinity away from the movie's genres (-ALPHA per genre present)

        A clamp to [0,1] keeps values in range without re-normalisation so
        multiple quick interactions accumulate naturally.
        """
        if not self._available:
            return None

        ALPHA = 0.12   # shift per interaction; ~8 likes to fully flip a genre

        current = self.get_user_features(user_id)
        if current is None:
            current = {
                "genre_affinity"  : [0.5] * 19,
                "top_genres"      : [],
                "favorite_decade" : movie_decade or 0,
                "avg_rating_given": 3.0,
                "watch_count"     : 0,
            }

        affinity  = np.array(current["genre_affinity"], dtype=np.float64)
        gv        = np.array(genre_vector, dtype=np.float64)
        direction = 1.0 if action == "like" else -1.0

        affinity  = np.clip(affinity + ALPHA * direction * gv, 0.0, 1.0)

        # EMA on avg_rating_given (like ≈ 4.5 stars, dislike ≈ 1.5 stars)
        target_rating            = 4.5 if action == "like" else 1.5
        current["avg_rating_given"] = 0.9 * current["avg_rating_given"] + 0.1 * target_rating
        current["watch_count"]      = current["watch_count"] + 1

        # Top genres from updated affinity
        top_idx              = np.argsort(affinity)[::-1][:3]
        current["genre_affinity"] = affinity.tolist()
        current["top_genres"]     = [GENRE_NAMES[i] for i in top_idx if affinity[i] > 0.45]

        self.set_user_features(user_id, current)
        return current

    def stats(self) -> dict:
        if not self._available:
            return {"available": False}
        info = self._client.info("keyspace")
        n_users = self._client.dbsize()
        return {"available": True, "total_keys": n_users, "keyspace": info}
