import pandas as pd
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

# Paths 
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")


def load_movies():
    movies = pd.read_parquet(os.path.join(PROCESSED_DIR, "movies_enriched.parquet"))
    print(f"  Loaded {len(movies):,} movies")
    return movies


def generate_embeddings(movies):
    # Load the pre-trained model
    model = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs={"low_cpu_mem_usage": True})

    # Get all the text strings we want to encode
    texts = movies["embedding_text"].tolist()

    print(f"  Encoding {len(texts):,} movies...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,  # normalize so we can use inner product = cosine similarity
    )

    print(f"  Generated embeddings with shape: {embeddings.shape}")
    print(f"  Each movie is now a vector of {embeddings.shape[1]} numbers")

    return model, embeddings


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    print(f"  Index built with {index.ntotal:,} vectors of dimension {dimension}")
    return index


def save_index(index, movies):
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, "faiss_index.bin")
    faiss.write_index(index, index_path)
    size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"  Saved faiss_index.bin ({size_mb:.1f} MB)")
    lookup = movies[["movieId", "clean_title", "year", "genres", "overview", "poster_url", "avg_rating", "num_ratings"]].to_dict("records")

    lookup_path = os.path.join(INDEX_DIR, "movie_lookup.pkl")
    with open(lookup_path, "wb") as f:
        pickle.dump(lookup, f)
    print(f"  Saved movie_lookup.pkl ({len(lookup):,} entries)")


def test_search(model, index, movies):
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

        query_vector = model.encode([query], normalize_embeddings=True)
        scores, indices = index.search(query_vector.astype(np.float32), k=10)

        print(f"\nTop 10 results for: '{query}'")

        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            movie = lookup[idx]
            year = f"({int(movie['year'])})" if pd.notna(movie['year']) else ""
            rating = f"⭐ {movie['avg_rating']:.1f}" if pd.notna(movie['avg_rating']) else ""
            print(f"  {rank:2d}. {movie['clean_title']} {year} — {movie['genres']} — {rating} — score: {score:.3f}")


def main():
    
    movies = load_movies()
    model, embeddings = generate_embeddings(movies)
    index = build_faiss_index(embeddings)
    save_index(index, movies)
    print("\n✓ Embedding generation complete!")
    test_search(model, index, movies)


if __name__ == "__main__":
    main()
