import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ml-25m")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

MIN_RATINGS_PER_MOVIE = 10  # drop movies with fewer ratings than this
MIN_RATINGS_PER_USER = 5    # drop users with fewer ratings than this


def load_raw_data():
    """Load raw CSVs into DataFrames."""
    print("Loading raw CSVs...")

    movies = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(RAW_DIR, "ratings.csv"))
    tags = pd.read_csv(os.path.join(RAW_DIR, "tags.csv"))

    print(f"  movies:  {len(movies):,} rows")
    print(f"  ratings: {len(ratings):,} rows")
    print(f"  tags:    {len(tags):,} rows")

    return movies, ratings, tags


def clean_movies(movies):
    """Clean movies DataFrame."""
    print("\nCleaning movies...")

    # Drop rows with missing title or genres
    movies = movies.dropna(subset=["title", "genres"])

    # Remove movies with "(no genres listed)"
    movies = movies[movies["genres"] != "(no genres listed)"]

    # Parse genres from "Action|Sci-Fi|Thriller" into ["Action", "Sci-Fi", "Thriller"]
    movies["genre_list"] = movies["genres"].str.split("|")

    # Extract year from title like "Toy Story (1995)"
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")

    # Clean title by removing the year suffix
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()

    print(f"  After cleaning: {len(movies):,} movies")
    return movies


def clean_ratings(ratings):
    """Clean ratings DataFrame."""
    print("\nCleaning ratings...")

    # Drop any rows with missing values
    ratings = ratings.dropna()

    # Convert timestamp to datetime
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")

    print(f"  After cleaning: {len(ratings):,} ratings")
    return ratings


def clean_tags(tags):
    """Clean tags DataFrame."""
    print("\nCleaning tags...")

    # Drop rows with missing tags
    tags = tags.dropna(subset=["tag"])

    # Lowercase all tags for consistency
    tags["tag"] = tags["tag"].str.lower().str.strip()

    print(f"  After cleaning: {len(tags):,} tags")
    return tags


def filter_low_activity(movies, ratings):
    """Remove movies and users with too few ratings."""
    print("\nFiltering low-activity movies and users...")

    # Count ratings per movie
    movie_counts = ratings.groupby("movieId").size()
    valid_movies = movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index
    ratings = ratings[ratings["movieId"].isin(valid_movies)]
    movies = movies[movies["movieId"].isin(valid_movies)]

    # Count ratings per user
    user_counts = ratings.groupby("userId").size()
    valid_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
    ratings = ratings[ratings["userId"].isin(valid_users)]

    print(f"  Movies remaining: {len(movies):,}")
    print(f"  Ratings remaining: {len(ratings):,}")
    print(f"  Users remaining: {ratings['userId'].nunique():,}")

    return movies, ratings


def compute_movie_stats(ratings):
    """Compute per-movie statistics."""
    print("\nComputing movie stats...")

    stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count"),
        rating_std=("rating", "std"),
    ).reset_index()

    # Fill std NaN (movies with 1 rating) with 0
    stats["rating_std"] = stats["rating_std"].fillna(0)

    print(f"  Stats computed for {len(stats):,} movies")
    return stats


def save_processed(movies, ratings, tags, movie_stats):
    """Save cleaned data to parquet files."""
    print(f"\nSaving to {PROCESSED_DIR}...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    movies.to_parquet(os.path.join(PROCESSED_DIR, "movies.parquet"), index=False)
    ratings.to_parquet(os.path.join(PROCESSED_DIR, "ratings.parquet"), index=False)
    tags.to_parquet(os.path.join(PROCESSED_DIR, "tags.parquet"), index=False)
    movie_stats.to_parquet(os.path.join(PROCESSED_DIR, "movie_stats.parquet"), index=False)

    print("  Done! Files saved:")
    for f in os.listdir(PROCESSED_DIR):
        size_mb = os.path.getsize(os.path.join(PROCESSED_DIR, f)) / (1024 * 1024)
        print(f"    {f} ({size_mb:.1f} MB)")


def main():
    print("=" * 50)
    print("CineRank - Data Preparation")
    print("=" * 50)

    # Load
    movies, ratings, tags = load_raw_data()

    # Clean
    movies = clean_movies(movies)
    ratings = clean_ratings(ratings)
    tags = clean_tags(tags)

    # Filter
    movies, ratings = filter_low_activity(movies, ratings)

    # Compute stats
    movie_stats = compute_movie_stats(ratings)

    # Save
    save_processed(movies, ratings, tags, movie_stats)

    print("\n✓ Data preparation complete!")


if __name__ == "__main__":
    main()