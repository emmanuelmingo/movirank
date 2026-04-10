"""
CineRank - Data Enrichment Script
Enriches MovieLens movies with TMDB plot overviews and poster URLs.

Setup:
    1. Sign up at https://www.themoviedb.org/ and get an API key
    2. Create a .env file in backend/ with: TMDB_API_KEY=your_key_here
    3. pip install requests python-dotenv
    4. Run: python backend/scripts/enrich_data.py

Note:
    - TMDB API has a rate limit of ~40 requests per second
    - This script processes 62K+ movies, so it takes a while (~30-45 min)
    - Progress is saved every 500 movies, so you can stop and resume safely
"""

import pandas as pd
import requests
import os
import time
import json
from dotenv import load_dotenv

# --- Load API key ---
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found. Create backend/.env with: TMDB_API_KEY=your_key_here")

# --- Paths ---
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CACHE_PATH = os.path.join(PROCESSED_DIR, "tmdb_cache.json")

# --- TMDB API ---
BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"


def load_cache():
    """Load cached TMDB results to avoid re-fetching."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            cache = json.load(f)
        print(f"  Loaded cache with {len(cache)} entries")
        return cache
    return {}


def save_cache(cache):
    """Save TMDB results to cache file."""
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def search_tmdb(title, year=None):
    """Search TMDB for a movie by title and optional year."""
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
    }
    if year and not pd.isna(year):
        params["year"] = int(year)

    try:
        response = requests.get(f"{BASE_URL}/search/movie", params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            movie = results[0]  # take the top match
            poster_path = movie.get("poster_path")
            return {
                "tmdb_id": movie.get("id"),
                "overview": movie.get("overview", ""),
                "poster_url": f"{POSTER_BASE_URL}{poster_path}" if poster_path else None,
                "tmdb_rating": movie.get("vote_average"),
                "tmdb_popularity": movie.get("popularity"),
            }
    except requests.exceptions.RequestException as e:
        print(f"    API error for '{title}': {e}")

    return {
        "tmdb_id": None,
        "overview": "",
        "poster_url": None,
        "tmdb_rating": None,
        "tmdb_popularity": None,
    }


def enrich_movies(movies):
    """Fetch TMDB data for all movies."""
    print("\nFetching TMDB data...")

    cache = load_cache()
    total = len(movies)
    results = []

    for i, row in movies.iterrows():
        movie_id = str(row["movieId"])

        # Use cache if available
        if movie_id in cache:
            results.append(cache[movie_id])
            continue

        # Fetch from TMDB
        tmdb_data = search_tmdb(row["clean_title"], row.get("year"))
        cache[movie_id] = tmdb_data
        results.append(tmdb_data)

        # Rate limiting: ~35 requests per second to stay safe
        time.sleep(0.03)

        # Progress + save checkpoint every 500 movies
        processed = len(results)
        if processed % 500 == 0:
            save_cache(cache)
            pct = (processed / total) * 100
            print(f"  {processed:,}/{total:,} ({pct:.1f}%) — cached")

    # Final cache save
    save_cache(cache)
    print(f"  Done! {len(results):,} movies processed")

    return pd.DataFrame(results)


def build_text_for_embeddings(row):
    """
    Combine title, genres, overview, and tags into a single text string.
    This is what sentence-transformers will encode on Day 3.
    """
    parts = []

    # Title
    parts.append(row["clean_title"])

    # Genres
    if isinstance(row.get("genre_list"), list):
        parts.append(", ".join(row["genre_list"]))

    # TMDB overview
    if row.get("overview"):
        parts.append(row["overview"])

    # User tags
    if row.get("tags_combined"):
        parts.append(row["tags_combined"])

    return " | ".join(parts)


def main():
    print("=" * 50)
    print("CineRank - Data Enrichment")
    print("=" * 50)

    # Load cleaned data from Day 1
    print("\nLoading cleaned data...")
    movies = pd.read_parquet(os.path.join(PROCESSED_DIR, "movies.parquet"))
    tags = pd.read_parquet(os.path.join(PROCESSED_DIR, "tags.parquet"))
    movie_stats = pd.read_parquet(os.path.join(PROCESSED_DIR, "movie_stats.parquet"))

    print(f"  {len(movies):,} movies loaded")

    # Aggregate tags per movie (combine all user tags into one string)
    print("\nAggregating tags per movie...")
    tags_per_movie = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: ", ".join(x.unique()))
        .reset_index()
        .rename(columns={"tag": "tags_combined"})
    )

    # Merge tags into movies
    movies = movies.merge(tags_per_movie, on="movieId", how="left")
    movies["tags_combined"] = movies["tags_combined"].fillna("")

    # Merge movie stats
    movies = movies.merge(movie_stats, on="movieId", how="left")

    # Fetch TMDB data
    tmdb_df = enrich_movies(movies)

    # Add TMDB columns to movies
    movies["tmdb_id"] = tmdb_df["tmdb_id"].values
    movies["overview"] = tmdb_df["overview"].values
    movies["poster_url"] = tmdb_df["poster_url"].values
    movies["tmdb_rating"] = tmdb_df["tmdb_rating"].values
    movies["tmdb_popularity"] = tmdb_df["tmdb_popularity"].values

    # Build combined text for embedding generation (Day 3)
    print("\nBuilding embedding text...")
    movies["embedding_text"] = movies.apply(build_text_for_embeddings, axis=1)

    # Save enriched data
    output_path = os.path.join(PROCESSED_DIR, "movies_enriched.parquet")
    movies.to_parquet(output_path, index=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Saved movies_enriched.parquet ({size_mb:.1f} MB)")
    print(f"  Columns: {list(movies.columns)}")
    print(f"  Movies with overviews: {movies['overview'].astype(bool).sum():,}/{len(movies):,}")
    print(f"  Movies with posters: {movies['poster_url'].notna().sum():,}/{len(movies):,}")


if __name__ == "__main__":
    main()
