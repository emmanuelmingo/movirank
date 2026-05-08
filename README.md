# Movirank

A movie recommendation system built on semantic search and personalised re-ranking. Describe what you want to watch and get results tailored to your taste.

## Features

- Semantic search using sentence embeddings and FAISS vector index
- Two-stage pipeline: vector retrieval followed by LightGBM re-ranking
- Per-user personalisation via a Redis feature store
- Like/dislike feedback updates your profile and reshuffles results in real time
- Built-in A/B framework comparing baseline (FAISS order) vs full ranker

## Prerequisites

- Python 3.11+
- Node.js 18+
- Redis
- [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/) extracted to `backend/data/raw/`
- TMDB API key (for poster enrichment)

## Getting Started

### Data pipeline

Run once from the project root in order:

```bash
python -m backend.scripts.prepare_data
python -m backend.scripts.enrich_data
python -m backend.scripts.build_features
python -m backend.scripts.build_embeddings
python -m backend.scripts.train_ranker
python -m backend.scripts.populate_feature_store
```

### Backend

```bash
python -m venv env
env\Scripts\activate
pip install fastapi "uvicorn[standard]" sentence-transformers faiss-cpu lightgbm numpy pandas redis pydantic scikit-learn requests
env\Scripts\uvicorn backend.app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

| Service  | Port |
|----------|------|
| Frontend | 5173 |
| Backend  | 8000 |

## Project Structure

```
movirank/
  backend/
    app/        # FastAPI app and Redis feature store client
    scripts/    # Data pipeline and model training scripts
    data/       # FAISS index, feature matrix, trained ranker, A/B logs
  frontend/
    src/        # React + Vite SPA
```

## Author

[Emmanuel Mingo](mailto:qqqwerrt1@gmail.com)
