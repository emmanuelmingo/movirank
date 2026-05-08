import os
import pickle
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_movies():
    path = os.path.join(PROCESSED_DIR, "movies_enriched.parquet")
    movies = pd.read_parquet(path)
    print(f"Loaded {len(movies):,} movies")
    return movies


def genre_multihot(movies):
    matrix = np.zeros((len(movies), len(GENRES)), dtype=np.float32)
    genre_index = {g: i for i, g in enumerate(GENRES)}
    for row, genre_list in enumerate(movies["genre_list"]):
        for g in genre_list:
            if g in genre_index:
                matrix[row, genre_index[g]] = 1.0
    print(f"  Genre multi-hot: {matrix.shape}")
    return matrix, [f"genre_{g.lower().replace('-', '_')}" for g in GENRES]


def minmax(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return np.zeros(len(series), dtype=np.float32)
    return ((series - lo) / (hi - lo)).to_numpy(dtype=np.float32)


def avg_rating_feature(movies):
    feat = minmax(movies["avg_rating"]).reshape(-1, 1)
    print(f"  Avg rating: min={movies['avg_rating'].min():.2f}  max={movies['avg_rating'].max():.2f}")
    return feat, ["avg_rating_norm"]


def rating_count_feature(movies):
    log_counts = np.log1p(movies["num_ratings"])
    feat = minmax(log_counts).reshape(-1, 1)
    print(f"  Rating count: min={movies['num_ratings'].min():,}  max={movies['num_ratings'].max():,}")
    return feat, ["rating_count_norm"]


def popularity_feature(movies):
    pop = movies["tmdb_popularity"].copy()
    missing_mask = pop.isna()
    pop = pop.fillna(0.0)
    log_pop = np.log1p(pop)
    feat = minmax(log_pop).reshape(-1, 1)
    print(f"  Popularity: {missing_mask.sum():,} nulls filled with 0")
    return feat, ["popularity_norm"]


def recency_feature(movies):
    year = movies["year"].copy()
    median_year = year.median()
    year = year.fillna(median_year)
    feat = minmax(year).reshape(-1, 1)
    print(f"  Recency: {movies['year'].isna().sum()} nulls filled with median ({int(median_year)})")
    return feat, ["recency_norm"]


def build_feature_matrix(movies):
    print("\nBuilding features...")
    genre_mat, genre_names = genre_multihot(movies)
    rating_mat, rating_names = avg_rating_feature(movies)
    count_mat, count_names = rating_count_feature(movies)
    pop_mat, pop_names = popularity_feature(movies)
    rec_mat, rec_names = recency_feature(movies)

    matrix = np.hstack([genre_mat, rating_mat, count_mat, pop_mat, rec_mat])
    names = genre_names + rating_names + count_names + pop_names + rec_names

    print(f"\nFeature matrix shape: {matrix.shape}")
    print(f"Features ({len(names)}): {names}")
    return matrix, names


def save(matrix, names, movie_ids):
    os.makedirs(INDEX_DIR, exist_ok=True)

    np.save(os.path.join(INDEX_DIR, "feature_matrix.npy"), matrix)
    np.save(os.path.join(INDEX_DIR, "feature_movie_ids.npy"), movie_ids)
    with open(os.path.join(INDEX_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(names, f)

    size_mb = matrix.nbytes / (1024 ** 2)
    print(f"\nSaved feature_matrix.npy      ({size_mb:.1f} MB)")
    print(f"Saved feature_movie_ids.npy   ({len(movie_ids):,} entries)")
    print(f"Saved feature_names.pkl       ({len(names)} features)")


def main():
    movies = load_movies()
    matrix, names = build_feature_matrix(movies)
    movie_ids = movies["movieId"].to_numpy(dtype=np.int64)
    save(matrix, names, movie_ids)

    idx = movies.index[movies["movieId"] == 1][0]
    row = matrix[idx]
    for name, val in zip(names, row):
        print(f"  {name:<25} {val:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
