# Movirank

A production-grade movie recommendation system built on a two-stage retrieval and re-ranking pipeline, personalised in real time using a Redis feature store. Search by describing what you want to watch — the ranker learns your taste as you like and dislike results.

---

## What makes it different

Most recommendation demos are one of two things: a cosine-similarity search that ignores who you are, or a collaborative-filter that ignores what you typed. Movirank combines both.

| Approach | Semantic relevance | Personalisation | Real-time feedback |
|---|---|---|---|
| Keyword / BM25 search | No | No | No |
| Pure collaborative filter | No | Yes | No |
| Vector similarity only | Yes | No | No |
| **Movirank** | **Yes** | **Yes** | **Yes** |

Additional things that set it apart:

- **LambdaRank, not heuristics.** The ranker is a LightGBM model trained on 1.89 million user-item pairs with graded relevance labels (rated 5★ → 2, 4★ → 1, unseen → 0). It optimises NDCG directly, which is what ranking quality actually means.
- **Two-stage pipeline.** FAISS retrieves 100 semantically relevant candidates in milliseconds; LightGBM then re-orders them using 27 features. You get speed *and* quality.
- **Incremental personalisation.** Liking or disliking a card nudges your genre affinity vector in Redis immediately. The next search uses the updated profile — no retraining, no batch job.
- **Built-in A/B framework.** Every search is assigned to a variant (baseline: FAISS order only, or full: LightGBM ranker) and logged to a JSONL file for offline analysis. The UI shows which variant is active.

---

## How it works

### 1. Offline data pipeline

```
MovieLens 25M ratings
        │
        ▼
  prepare_data.py        — joins movies, tags, links; cleans titles; parses years
        │
        ▼
  enrich_data.py         — fetches TMDB posters, overviews, vote counts
        │
        ▼
  build_features.py      — 23 per-movie features:
                             19 genre multi-hot vectors
                              + avg_rating_norm  (mean rating, log-normalised)
                              + rating_count_norm (popularity proxy)
                              + popularity_norm  (TMDB vote count, log-normalised)
                              + recency_norm     (release year, min-max scaled)
        │
        ▼
  build_embeddings.py    — encodes title + genres with all-MiniLM-L6-v2 (384 dims)
                           builds FAISS IndexFlatIP on L2-normalised vectors
```

### 2. Ranker training

```
train_ranker.py

  5,000 users sampled (≥ 20 ratings each)
  Positive pairs  : movies rated ≥ 4★  → relevance label 2 (5★) or 1 (4★)
  Negative samples: unseen movies       → relevance label 0, 4:1 ratio

  Feature vector (27 dims):
    Item features (23)   : genre multi-hot + rating stats + popularity + recency
    User features (4)    : genre_overlap with user affinity · avg_rating_norm
                           watch_count_norm · decade_match

  Model: LightGBM LambdaRank
    objective      : lambdarank
    ndcg_eval_at   : [5, 10]
    num_leaves     : 63
    learning_rate  : 0.05
    n_estimators   : 500

  Offline results:
    NDCG@10 : 0.7889
    MRR@10  : 0.9925
```

### 3. Query-time pipeline

```
User query: "gritty crime drama set in New York"
        │
        ▼
  Sentence embedding (all-MiniLM-L6-v2, 384 dims)
        │
        ▼
  FAISS IndexFlatIP  →  top-100 candidates by cosine similarity
        │
        ├── Baseline variant (A)
        │     Return top-k in FAISS order  ─────────────────────────────┐
        │                                                                │
        └── Ranker variant (B)                                           │
              Pull user profile from Redis (genre affinity,             │
              avg rating, watch count, favourite decade)                 │
              Build 27-feature matrix for all 100 candidates            │
              LightGBM.predict() → re-order → top-k               ──────┤
                                                                        │
        ◄───────────────────────────────────────────────────────────────┘
  Return ranked results + variant label + log event to JSONL
```

### 4. User profile (Redis feature store)

Each named user has a profile stored as a Redis Hash (`user:{id}`):

| Field | Type | Description |
|---|---|---|
| `genre_affinity` | float[19] | Weighted preference per genre, range [0, 1] |
| `top_genres` | string[3] | Top 3 genre names by affinity score |
| `favorite_decade` | int | Decade with most liked films (e.g. 1990) |
| `avg_rating_given` | float | Mean rating the user gives (EMA-updated) |
| `watch_count` | int | Total interactions |

**Incremental update on feedback:**

```
Like    → affinity += 0.12 × genre_vector   (clipped to 1.0)
Dislike → affinity -= 0.12 × genre_vector   (clipped to 0.0)
avg_rating ← 0.9 × current + 0.1 × (4.5 if like else 1.5)
```

~8 interactions fully flip a genre from neutral to strong preference. No retraining required.

---

## Project structure

```
movirank/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app — search, feedback, A/B logging
│   │   └── feature_store.py   # Redis client — user/item features, feedback updates
│   ├── scripts/
│   │   ├── prepare_data.py    # Clean and join MovieLens data
│   │   ├── enrich_data.py     # Fetch TMDB metadata and posters
│   │   ├── build_features.py  # Build 23-feature matrix per movie
│   │   ├── build_embeddings.py# Sentence embeddings + FAISS index
│   │   ├── train_ranker.py    # Train LightGBM LambdaRank model
│   │   └── populate_feature_store.py  # Seed Redis with user/item profiles
│   └── data/
│       ├── index/             # faiss_index.bin, feature_matrix.npy, movie_lookup.pkl
│       ├── models/            # ranker.lgb
│       └── ab_logs/           # events.jsonl (A/B event log)
└── frontend/
    └── src/
        ├── App.jsx
        └── components/
            ├── SearchBar.jsx
            ├── MovieCard.jsx  # Like/dislike buttons, poster, genre tags, stars
            ├── UserSwitcher.jsx
            └── Toast.jsx
```

---

## Running locally

### Prerequisites

- Python 3.11+
- Node.js 18+
- Redis (see below)
- A [TMDB API key](https://www.themoviedb.org/settings/api) (only needed to re-run enrichment)
- [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/) extracted to `backend/data/raw/`

### 1. Python environment

```bash
cd movirank
python -m venv env
env\Scripts\activate          # Windows
# source env/bin/activate     # macOS / Linux

pip install fastapi uvicorn[standard] sentence-transformers faiss-cpu \
            lightgbm numpy pandas redis pydantic scikit-learn requests
```

### 2. Build the index and train the ranker

Run these scripts once in order from the project root:

```bash
python -m backend.scripts.prepare_data
python -m backend.scripts.enrich_data        # needs TMDB_API_KEY in .env
python -m backend.scripts.build_features
python -m backend.scripts.build_embeddings
python -m backend.scripts.train_ranker
```

Each script saves its output to `backend/data/` so you only need to re-run if the data changes.

### 3. Start Redis

**Windows (WSL):**
```bash
wsl
sudo service redis-server start
```

**macOS:**
```bash
brew install redis && brew services start redis
```

**Linux:**
```bash
sudo apt install redis-server && sudo service redis-server start
```

### 4. Seed the feature store

Precomputes user profiles for all users with ≥ 20 ratings and writes item features for all 24k movies:

```bash
python -m backend.scripts.populate_feature_store
```

This takes a few minutes. Without it, the system still works but all users fall back to the anonymous (genre-inferred) profile.

### 5. Start the backend

From the project root:

```bash
env\Scripts\uvicorn backend.app.main:app --reload
```

The API is now available at `http://127.0.0.1:8000`.

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?q=&k=10&user_id=` | Two-stage search. Returns `{variant, results}` |
| `POST` | `/user/{id}/feedback` | Submit like/dislike. Updates Redis profile immediately |
| `GET` | `/user/{id}/features` | Inspect a user's current Redis profile |
| `GET` | `/ab/summary` | Aggregate A/B event counts and top queries |
| `GET` | `/feature-store/stats` | Redis key counts and keyspace info |

### Search response

```json
{
  "variant": "ranker",
  "results": [
    {
      "movieId": 593,
      "clean_title": "Silence of the Lambs, The",
      "genres": "Crime|Horror|Thriller",
      "year": 1991,
      "avg_rating": 4.18,
      "poster_url": "https://image.tmdb.org/...",
      "score": 0.8421
    }
  ]
}
```

### Feedback body

```json
{ "movie_id": 593, "action": "like" }
```

---

## A/B testing

Every search is assigned a variant:

- **A — Baseline:** Results returned in raw FAISS cosine-similarity order. No ranker.
- **B — Ranker:** Full LightGBM re-ranking with user and item features.

Assignment is stable per user (MD5 hash of `user_id`) so the same user always sees the same variant. Anonymous sessions are split randomly per request.

Events are appended to `backend/data/ab_logs/events.jsonl`:

```json
{"ts": "2026-05-07T14:22:01Z", "user_id": 42, "query": "sci-fi adventure", "variant": "ranker", "top_movie_ids": [260, 1196, 1210]}
```

Check live stats:

```bash
curl http://127.0.0.1:8000/ab/summary
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector index | FAISS `IndexFlatIP` (exact cosine similarity) |
| Re-ranker | LightGBM LambdaRank |
| Feature store | Redis (7-day TTL, hash per user/item) |
| API | FastAPI + Uvicorn |
| Frontend | React 19 + Vite + Tailwind CSS |
| Dataset | MovieLens 25M + TMDB enrichment |
