import json
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.app.feature_store import FeatureStore, GENRE_NAMES

BASE          = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE, "..", "data", "processed")
INDEX_DIR     = os.path.join(BASE, "..", "data", "index")

MIN_RATINGS   = 20
BATCH_SIZE    = 500   # users written per Redis pipeline flush


def load_assets():
    ratings = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "ratings.parquet"),
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "movies_enriched.parquet"),
        columns=["movieId", "year"],
    )
    feature_matrix    = np.load(os.path.join(INDEX_DIR, "feature_matrix.npy"))
    feature_movie_ids = np.load(os.path.join(INDEX_DIR, "feature_movie_ids.npy"))
    print(f"  Ratings  : {len(ratings):,}")
    print(f"  Movies   : {len(movies):,}")
    print(f"  Features : {feature_matrix.shape}")
    return ratings, movies, feature_matrix, feature_movie_ids


# Item features 

def populate_items(fs: FeatureStore, feature_matrix: np.ndarray, feature_movie_ids: np.ndarray):
    print("\nPopulating item features...")
    t = time.time()
    genre_matrix = feature_matrix[:, :19]

    pipe = fs._client.pipeline(transaction=False)
    for feat_idx, movie_id in enumerate(feature_movie_ids):
        row   = feature_matrix[feat_idx]
        key   = f"item:{int(movie_id)}"
        pipe.hset(key, mapping={
            "genre_vector"     : json.dumps(genre_matrix[feat_idx].tolist()),
            "avg_rating_norm"  : float(row[19]),
            "rating_count_norm": float(row[20]),
            "popularity_norm"  : float(row[21]),
            "recency_norm"     : float(row[22]),
            "feat_idx"         : feat_idx,
        })
        pipe.expire(key, 7 * 24 * 3600)

        if feat_idx % 2000 == 1999:
            pipe.execute()
            pipe = fs._client.pipeline(transaction=False)
            print(f"  {feat_idx + 1:,} / {len(feature_movie_ids):,}")

    pipe.execute()
    print(f"  Done: {len(feature_movie_ids):,} items in {time.time() - t:.1f}s")


#  User features 

def compute_user_features(user_ratings: pd.DataFrame, movie_year_map: dict,genre_matrix: np.ndarray, movie_to_feat_idx: dict) -> dict:
    user_ratings = user_ratings.copy()
    user_ratings["weight"] = user_ratings["rating"] - 3.0

    # weighted sum of genre vectors, normalised to unit scale
    genre_affinity = np.zeros(19, dtype=np.float64)
    decade_weights: dict[int, float] = {}

    for _, row in user_ratings.iterrows():
        mid = int(row["movieId"])
        if mid not in movie_to_feat_idx:
            continue
        fi     = movie_to_feat_idx[mid]
        weight = row["weight"]
        genre_affinity += weight * genre_matrix[fi]

        # Decade affinity (weight positive ratings only)
        if weight > 0 and mid in movie_year_map:
            decade = (int(movie_year_map[mid]) // 10) * 10
            decade_weights[decade] = decade_weights.get(decade, 0.0) + weight

    # Normalise genre affinity to [0,1]
    norm = np.abs(genre_affinity).max()
    if norm > 0:
        genre_affinity = ((genre_affinity / norm) + 1) / 2   # shift to [0,1]

    # Top genres by affinity
    top_idx   = np.argsort(genre_affinity)[::-1][:3]
    top_genres = [GENRE_NAMES[i] for i in top_idx if genre_affinity[i] > 0.5]

    # Favorite decade
    favorite_decade = max(decade_weights, key=decade_weights.get) if decade_weights else 0

    return {
        "genre_affinity"  : genre_affinity.tolist(),
        "top_genres"      : top_genres,
        "favorite_decade" : favorite_decade,
        "avg_rating_given": float(user_ratings["rating"].mean()),
        "watch_count"     : len(user_ratings),
    }


def populate_users(fs: FeatureStore, ratings: pd.DataFrame, movies: pd.DataFrame, genre_matrix: np.ndarray, feature_movie_ids: np.ndarray):
    t = time.time()

    movie_to_feat_idx = {int(mid): i for i, mid in enumerate(feature_movie_ids)}
    movie_year_map    = movies.dropna(subset=["year"]).set_index("movieId")["year"].to_dict()

    counts    = ratings.groupby("userId").size()
    eligible  = counts[counts >= MIN_RATINGS].index
    print(f"  Eligible users: {len(eligible):,}")

    grouped = ratings[ratings["userId"].isin(eligible)].groupby("userId")

    done = 0
    pipe = fs._client.pipeline(transaction=False)

    for user_id, user_df in grouped:
        feats = compute_user_features(user_df, movie_year_map, genre_matrix, movie_to_feat_idx)
        key   = f"user:{int(user_id)}"
        pipe.hset(key, mapping={
            "genre_affinity"  : json.dumps(feats["genre_affinity"]),
            "top_genres"      : json.dumps(feats["top_genres"]),
            "favorite_decade" : feats["favorite_decade"],
            "avg_rating_given": feats["avg_rating_given"],
            "watch_count"     : feats["watch_count"],
        })
        pipe.expire(key, 7 * 24 * 3600)
        done += 1

        if done % BATCH_SIZE == 0:
            pipe.execute()
            pipe = fs._client.pipeline(transaction=False)
            elapsed = time.time() - t
            rate    = done / elapsed
            print(f"  {done:,} / {len(eligible):,}  ({rate:.0f} users/s)")

    pipe.execute()
    print(f"  Done: {done:,} users in {time.time() - t:.1f}s")


# Main

def main():

    fs = FeatureStore()
    if not fs.available:
        print("\nERROR: Redis is not reachable. Start Redis first:")
        print("  WSL:    sudo service redis-server start")
        print("  Docker: docker run -d -p 6379:6379 redis:7-alpine")
        return

    ratings, movies, feature_matrix, feature_movie_ids = load_assets()
    genre_matrix = feature_matrix[:, :19]

    populate_items(fs, feature_matrix, feature_movie_ids)
    populate_users(fs, ratings, movies, genre_matrix, feature_movie_ids)

    stats = fs.stats()
    print(f"\nFeature store stats: {stats}")
    print("\nDone.")


if __name__ == "__main__":
    main()
