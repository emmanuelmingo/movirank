"""
CineRank - Embedding Generation & FAISS Index Builder
Generates embeddings for all movies and builds a searchable FAISS index.

Setup:
    pip install sentence-transformers faiss-cpu

Usage:
    python backend/scripts/build_embeddings.py

What this script does (step by step):
    1. Loads the enriched movie data from Day 2
    2. Uses a pre-trained sentence-transformers model to convert each movie's
       text (title + genres + overview + tags) into a 384-dimensional vector
    3. Stores all vectors in a FAISS index for fast similarity search
    4. Saves the index and a mapping file so your FastAPI app can load them
    5. Lets you test searches from the terminal
"""

import pandas as pd
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

# --- Paths ---
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")


def load_movies():
    """
    Step 1: Load the enriched movie data from Day 2.

    This parquet file has everything: title, genres, overview, tags, poster URLs,
    and most importantly the 'embedding_text' column — the combined text we'll
    convert into vectors.
    """
    print("Step 1: Loading enriched movie data...")
    movies = pd.read_parquet(os.path.join(PROCESSED_DIR, "movies_enriched.parquet"))
    print(f"  Loaded {len(movies):,} movies")
    return movies


def generate_embeddings(movies):
    """
    Step 2: Convert each movie's text into a vector (embedding).

    How it works:
    - We load a pre-trained model called 'all-MiniLM-L6-v2'
      This model was trained on millions of text pairs to understand meaning.
      It's small (80MB) but powerful — good balance of speed and quality.

    - For each movie, the model reads the embedding_text like:
      "Inception | Sci-Fi, Action, Thriller | A thief who steals corporate
       secrets through dream-sharing technology..."

    - It outputs a vector of 384 numbers. These numbers encode the MEANING
      of that text. Movies with similar plots/genres/themes will have
      vectors that are close together in 384-dimensional space.

    - Example: The vectors for "Alien" and "Aliens" will be very close.
      The vectors for "Alien" and "Toy Story" will be far apart.
      The vector for "Alien" and "The Thing" will be moderately close
      (both are sci-fi horror about isolated people fighting creatures).

    Why 'all-MiniLM-L6-v2'?
    - 384 dimensions (small enough to be fast, big enough to capture meaning)
    - Encodes at ~2000 sentences/second on CPU
    - Top performer on semantic similarity benchmarks for its size
    """
    print("\nStep 2: Generating embeddings...")
    print("  Loading sentence-transformers model (first run downloads ~80MB)...")

    # Load the pre-trained model
    model = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs={"low_cpu_mem_usage": True})

    # Get all the text strings we want to encode
    texts = movies["embedding_text"].tolist()

    print(f"  Encoding {len(texts):,} movies...")
    print("  This takes 2-5 minutes on CPU...")

    # model.encode() does the heavy lifting:
    # - Tokenizes each text (splits into subwords the model understands)
    # - Passes tokens through 6 transformer layers
    # - Pools the output into a single 384-dim vector per text
    # - show_progress_bar=True gives us a progress bar
    # - batch_size=64 means it processes 64 movies at a time (faster than one by one)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,  # normalize so we can use inner product = cosine similarity
    )

    # embeddings is now a numpy array of shape (24000, 384)
    # Each row is one movie's vector
    print(f"  Generated embeddings with shape: {embeddings.shape}")
    print(f"  Each movie is now a vector of {embeddings.shape[1]} numbers")

    return model, embeddings


def build_faiss_index(embeddings):
    """
    Step 3: Build a FAISS index from the embeddings.

    What is FAISS?
    - FAISS = Facebook AI Similarity Search
    - It's a library that stores vectors and lets you find the closest ones
      to any query vector, extremely fast (sub-millisecond for 24K movies)

    How it works:
    - We use IndexFlatIP (Inner Product). Since our embeddings are normalized,
      inner product = cosine similarity. Higher score = more similar.

    - When you search, FAISS compares your query vector against all 24K movie
      vectors and returns the top K most similar ones.

    - For 24K movies, a flat (brute-force) index is fine. FAISS also has
      approximate indexes for millions of vectors, but we don't need that here.

    Why not just use numpy?
    - You could! np.dot(query, all_embeddings.T) would work.
    - But FAISS is optimized with SIMD instructions, uses less memory,
      and scales to millions of vectors. Good habit to learn.
    """
    print("\nStep 3: Building FAISS index...")

    # Get the dimension of our vectors (384)
    dimension = embeddings.shape[1]

    # Create a flat index using inner product (cosine similarity since vectors are normalized)
    # IndexFlatIP = "Flat index, Inner Product"
    # Flat = brute force (compares against every vector). Fast enough for 24K movies.
    index = faiss.IndexFlatIP(dimension)

    # Add all movie vectors to the index
    # After this, each vector gets an ID (0, 1, 2, ... 23999)
    # These IDs correspond to row positions in our movies DataFrame
    index.add(embeddings.astype(np.float32))

    print(f"  Index built with {index.ntotal:,} vectors of dimension {dimension}")

    return index


def save_index(index, movies):
    """
    Step 4: Save the FAISS index and movie ID mapping.

    We save two files:
    - faiss_index.bin: The FAISS index (all vectors, ready to search)
    - movie_lookup.pkl: A mapping from FAISS index position → movie data
      so when FAISS returns "result #7842", we can look up that it's "Inception"
    """
    print("\nStep 4: Saving index and lookup...")
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Save FAISS index
    index_path = os.path.join(INDEX_DIR, "faiss_index.bin")
    faiss.write_index(index, index_path)
    size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"  Saved faiss_index.bin ({size_mb:.1f} MB)")

    # Save movie lookup — a list of dicts, one per movie, in the same order as the FAISS index
    # This is how we translate FAISS results back to movie info
    lookup = movies[["movieId", "clean_title", "year", "genres", "overview", "poster_url",
                      "avg_rating", "num_ratings"]].to_dict("records")

    lookup_path = os.path.join(INDEX_DIR, "movie_lookup.pkl")
    with open(lookup_path, "wb") as f:
        pickle.dump(lookup, f)
    print(f"  Saved movie_lookup.pkl ({len(lookup):,} entries)")


def test_search(model, index, movies):
    """
    Step 5: Interactive search from the terminal.

    How search works:
    1. You type a query like "dark psychological thriller with a twist"
    2. The SAME sentence-transformers model encodes your query into a 384-dim vector
    3. FAISS compares that vector against all 24K movie vectors
    4. It returns the top 10 closest matches (highest cosine similarity)
    5. We look up the movie info and display it

    This is the core of semantic search:
    - You don't need exact keyword matches
    - "space adventure" finds "Interstellar" even if those words aren't in the overview
    - The model understands MEANING, not just words
    """
    print("\n" + "=" * 50)
    print("SEMANTIC SEARCH TEST")
    print("Type a query to search for movies. Type 'quit' to exit.")
    print("=" * 50)

    lookup = movies[["clean_title", "year", "genres", "avg_rating", "num_ratings"]].to_dict("records")

    while True:
        query = input("\n🔍 Search: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        # Encode the query using the same model
        # This produces a vector in the same 384-dim space as our movie vectors
        query_vector = model.encode([query], normalize_embeddings=True)

        # Search the FAISS index
        # k=10 means return top 10 results
        # scores = cosine similarity scores (higher = more similar, max 1.0)
        # indices = positions in the FAISS index (which map to rows in our DataFrame)
        scores, indices = index.search(query_vector.astype(np.float32), k=10)

        print(f"\nTop 10 results for: '{query}'")
        print("-" * 60)

        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            movie = lookup[idx]
            year = f"({int(movie['year'])})" if pd.notna(movie['year']) else ""
            rating = f"⭐ {movie['avg_rating']:.1f}" if pd.notna(movie['avg_rating']) else ""
            print(f"  {rank:2d}. {movie['clean_title']} {year} — {movie['genres']} — {rating} — score: {score:.3f}")


def main():
    print("=" * 50)
    print("CineRank - Embedding Generation & Index Builder")
    print("=" * 50)

    # Step 1: Load data
    movies = load_movies()

    # Step 2: Generate embeddings
    model, embeddings = generate_embeddings(movies)

    # Step 3: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 4: Save everything
    save_index(index, movies)

    print("\n✓ Embedding generation complete!")

    # Step 5: Test it out
    test_search(model, index, movies)


if __name__ == "__main__":
    main()
