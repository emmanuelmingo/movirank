"""
CineRank - LightGBM Ranker Training Pipeline

Steps:
  1. Load ratings + feature matrix
  2. Sample users; build positive/negative pairs
  3. Compute per-pair genre-overlap feature
  4. Train/val/test split (by user, so no user leaks across splits)
  5. Train LightGBM LambdaRank model
  6. Evaluate: NDCG@5, NDCG@10, MRR@10
  7. Save model + metadata

Positive label  : rating >= 4.0  →  graded: 5★ = 2, 4★ = 1
Negative label  : 0  (unrated movies sampled per user)
Negative ratio  : 4 negatives per positive

Usage:
    python backend/scripts/train_ranker.py
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE, "..", "data", "processed")
INDEX_DIR     = os.path.join(BASE, "..", "data", "index")
MODEL_DIR     = os.path.join(BASE, "..", "data", "models")

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_USERS       = 5_000   # users to sample for train+val+test
MIN_RATINGS   = 20      # skip users with fewer ratings
POS_THRESHOLD = 4.0     # rating >= this is a positive
NEG_RATIO     = 4       # negatives per positive per user
VAL_FRAC      = 0.10    # fraction of sampled users → validation
TEST_FRAC     = 0.20    # fraction of sampled users → test
SEED          = 42


# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_data():
    print("Step 1: Loading data...")
    t = time.time()

    ratings = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "ratings.parquet"),
        columns=["userId", "movieId", "rating"],
    )
    movies = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "movies_enriched.parquet"),
        columns=["movieId", "year"],
    )
    feature_matrix    = np.load(os.path.join(INDEX_DIR, "feature_matrix.npy"))
    feature_movie_ids = np.load(os.path.join(INDEX_DIR, "feature_movie_ids.npy"))
    with open(os.path.join(INDEX_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    print(f"  Ratings        : {len(ratings):>12,}")
    print(f"  Unique users   : {ratings['userId'].nunique():>12,}")
    print(f"  Feature matrix : {feature_matrix.shape}")
    print(f"  Loaded in {time.time() - t:.1f}s")
    return ratings, movies, feature_matrix, feature_movie_ids, feature_names


# ── 2. Sample users ────────────────────────────────────────────────────────────

def sample_users(ratings, n, min_ratings, seed):
    print(f"\nStep 2: Sampling up to {n:,} users with >= {min_ratings} ratings...")
    counts = ratings.groupby("userId").size()
    eligible = counts[counts >= min_ratings].index.to_numpy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    print(f"  Eligible: {len(eligible):,}  ->  Sampled: {len(chosen):,}")
    return chosen


# ── 3. Build training pairs ────────────────────────────────────────────────────

def build_pairs(ratings, movies, feature_matrix, feature_movie_ids, sampled_users, seed):
    """
    Returns a DataFrame with columns:
      userId, movieId, label, feat_idx,
      genre_overlap, user_avg_rating_norm, user_watch_count_norm, user_decade_match
    """
    print("\nStep 3: Building training pairs...")
    t = time.time()
    rng = np.random.default_rng(seed)

    movie_to_idx   = {int(mid): i for i, mid in enumerate(feature_movie_ids)}
    feat_movie_arr = feature_movie_ids
    genre_matrix   = feature_matrix[:, :19]
    user_set       = set(sampled_users.tolist())

    # Movie decade map: movieId -> decade (e.g. 1994 -> 1990)
    movie_decade_map = {
        int(r.movieId): (int(r.year) // 10) * 10
        for r in movies.dropna(subset=["year"]).itertuples()
    }

    # Filter ratings once
    rat = ratings[
        ratings["userId"].isin(user_set) &
        ratings["movieId"].isin(movie_to_idx)
    ].copy()

    # Graded labels
    rat["label"] = 0
    rat.loc[rat["rating"] >= 5.0, "label"] = 2
    rat.loc[(rat["rating"] >= 4.0) & (rat["rating"] < 5.0), "label"] = 1

    pos = rat[rat["rating"] >= POS_THRESHOLD][["userId", "movieId", "label"]].copy()
    pos["feat_idx"] = pos["movieId"].map(movie_to_idx).astype(int)

    # Per-user stats computed once from all their ratings
    user_stats = {}
    for uid, group in rat.groupby("userId"):
        pos_mids = group.loc[group["rating"] >= POS_THRESHOLD, "movieId"]
        decades  = [movie_decade_map[int(m)] for m in pos_mids if int(m) in movie_decade_map]
        fav_decade = max(set(decades), key=decades.count) if decades else 0
        user_stats[uid] = {
            "avg_rating_norm"     : (group["rating"].mean() - 0.5) / 4.5,
            "watch_count_norm"    : min(np.log1p(len(group)) / np.log1p(10_000), 1.0),
            "favorite_decade"     : fav_decade,
            "genre_affinity"      : genre_matrix[
                [movie_to_idx[int(m)] for m in pos_mids if int(m) in movie_to_idx]
            ].mean(axis=0) if len(pos_mids) else np.zeros(19),
        }

    # Vectorised user features for positives
    pos_stats = [user_stats[uid] for uid in pos["userId"]]
    pos_user_vecs  = np.stack([s["genre_affinity"] for s in pos_stats])
    pos_movie_vecs = genre_matrix[pos["feat_idx"].to_numpy()]
    pos["genre_overlap"]          = (pos_user_vecs * pos_movie_vecs).sum(axis=1)
    pos["user_avg_rating_norm"]   = [s["avg_rating_norm"]  for s in pos_stats]
    pos["user_watch_count_norm"]  = [s["watch_count_norm"] for s in pos_stats]
    pos["user_decade_match"]      = [
        1.0 if movie_decade_map.get(int(mid), -1) == user_stats[uid]["favorite_decade"] else 0.0
        for uid, mid in zip(pos["userId"], pos["movieId"])
    ]

    # Negatives: sample unseen movies per user
    rated_by_user     = rat.groupby("userId")["movieId"].apply(set)
    pos_count_by_user = pos.groupby("userId").size()

    neg_rows = []
    for uid in sampled_users:
        if uid not in pos_count_by_user or uid not in user_stats:
            continue
        seen   = rated_by_user.get(uid, set())
        unseen = feat_movie_arr[~np.isin(feat_movie_arr, list(seen))]
        n_neg  = min(int(pos_count_by_user[uid]) * NEG_RATIO, len(unseen))
        if n_neg == 0:
            continue
        stats = user_stats[uid]
        for mid in rng.choice(unseen, size=n_neg, replace=False):
            fi      = movie_to_idx[int(mid)]
            overlap = float(np.dot(stats["genre_affinity"], genre_matrix[fi]))
            decade_match = 1.0 if movie_decade_map.get(int(mid), -1) == stats["favorite_decade"] else 0.0
            neg_rows.append((uid, int(mid), 0, fi, overlap,
                             stats["avg_rating_norm"], stats["watch_count_norm"], decade_match))

    neg_df = pd.DataFrame(neg_rows, columns=[
        "userId", "movieId", "label", "feat_idx",
        "genre_overlap", "user_avg_rating_norm", "user_watch_count_norm", "user_decade_match",
    ])

    df = pd.concat([pos, neg_df], ignore_index=True).sort_values("userId").reset_index(drop=True)
    print(f"  Pairs: {len(df):,}  (pos={(df['label']>0).sum():,}, neg={(df['label']==0).sum():,})")
    print(f"  Built in {time.time() - t:.1f}s")
    return df


# ── 4. Split by user ───────────────────────────────────────────────────────────

def user_split(df, sampled_users, val_frac, test_frac, seed):
    print("\nStep 4: Splitting train / val / test by user...")
    rng = np.random.default_rng(seed)
    users = sampled_users.copy()
    rng.shuffle(users)

    n       = len(users)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)

    test_users  = set(users[:n_test].tolist())
    val_users   = set(users[n_test: n_test + n_val].tolist())
    train_users = set(users[n_test + n_val:].tolist())

    train_df = df[df["userId"].isin(train_users)]
    val_df   = df[df["userId"].isin(val_users)]
    test_df  = df[df["userId"].isin(test_users)]

    print(f"  Train users: {len(train_users):,}  rows: {len(train_df):,}")
    print(f"  Val   users: {len(val_users):,}  rows: {len(val_df):,}")
    print(f"  Test  users: {len(test_users):,}  rows: {len(test_df):,}")
    return train_df, val_df, test_df


# ── 5. Build LightGBM arrays ───────────────────────────────────────────────────

def to_lgb_arrays(df, feature_matrix, all_feat_names):
    """Returns X, y, group sizes (sorted by userId)."""
    df = df.sort_values("userId")
    item_feats   = feature_matrix[df["feat_idx"].to_numpy()]
    user_feats   = df[["genre_overlap", "user_avg_rating_norm",
                        "user_watch_count_norm", "user_decade_match"]].to_numpy(dtype=np.float32)
    X      = np.hstack([item_feats, user_feats]).astype(np.float32)
    y      = df["label"].to_numpy(dtype=np.int32)
    groups = df.groupby("userId", sort=True).size().to_numpy(dtype=np.int32)
    return X, y, groups


# ── 6. Train ───────────────────────────────────────────────────────────────────

def train(X_tr, y_tr, g_tr, X_val, y_val, g_val, feat_names):
    print("\nStep 5: Training LightGBM LambdaRank...")

    train_ds = lgb.Dataset(X_tr,  label=y_tr,  group=g_tr,  feature_name=feat_names)
    val_ds   = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_ds)

    params = {
        "objective"       : "lambdarank",
        "metric"          : "ndcg",
        "ndcg_eval_at"    : [5, 10],
        "learning_rate"   : 0.05,
        "num_leaves"      : 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq"    : 5,
        "verbose"         : -1,
        "seed"            : SEED,
    }

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=500,
        valid_sets=[val_ds],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    return model


# ── 7. Evaluate ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, groups_test, k=10):
    print(f"\nStep 6: Evaluating on test set...")
    scores = model.predict(X_test)

    ndcg5_scores, ndcg10_scores, mrr_scores = [], [], []
    start = 0
    for g in groups_test:
        end = start + g
        y_g, s_g = y_test[start:end], scores[start:end]
        start = end

        if y_g.max() == 0:
            continue

        ndcg5_scores.append(ndcg_score([y_g], [s_g], k=5))
        ndcg10_scores.append(ndcg_score([y_g], [s_g], k=10))

        ranked = y_g[np.argsort(s_g)[::-1]][:k]
        hit = np.where(ranked > 0)[0]
        mrr_scores.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)

    print(f"  NDCG@5  : {np.mean(ndcg5_scores):.4f}")
    print(f"  NDCG@10 : {np.mean(ndcg10_scores):.4f}")
    print(f"  MRR@10  : {np.mean(mrr_scores):.4f}")

    print("\n  Feature importances (gain):")
    names = model.feature_name()
    gains = model.feature_importance(importance_type="gain")
    for name, gain in sorted(zip(names, gains), key=lambda x: -x[1])[:10]:
        print(f"    {name:<25}  {gain:,.0f}")

    return np.mean(ndcg5_scores), np.mean(ndcg10_scores), np.mean(mrr_scores)


# ── 8. Save ────────────────────────────────────────────────────────────────────

def save_model(model, feat_names):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "ranker.lgb")
    meta_path  = os.path.join(MODEL_DIR, "ranker_meta.pkl")

    model.save_model(model_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"feature_names": feat_names, "n_features": len(feat_names)}, f)

    size_kb = os.path.getsize(model_path) / 1024
    print(f"\n  Saved ranker.lgb     ({size_kb:.0f} KB)")
    print(f"  Saved ranker_meta.pkl")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("CineRank - LightGBM Ranker Training")
    print("=" * 55)

    ratings, movies, feature_matrix, feature_movie_ids, feature_names = load_data()
    sampled_users = sample_users(ratings, N_USERS, MIN_RATINGS, SEED)

    df = build_pairs(ratings, movies, feature_matrix, feature_movie_ids, sampled_users, SEED)

    train_df, val_df, test_df = user_split(df, sampled_users, VAL_FRAC, TEST_FRAC, SEED)

    all_feat_names = feature_names + [
        "genre_overlap", "user_avg_rating_norm", "user_watch_count_norm", "user_decade_match",
    ]
    X_tr,  y_tr,  g_tr  = to_lgb_arrays(train_df, feature_matrix, all_feat_names)
    X_val, y_val, g_val  = to_lgb_arrays(val_df,   feature_matrix, all_feat_names)
    X_te,  y_te,  g_te   = to_lgb_arrays(test_df,  feature_matrix, all_feat_names)

    model = train(X_tr, y_tr, g_tr, X_val, y_val, g_val, all_feat_names)
    evaluate(model, X_te, y_te, g_te)
    save_model(model, all_feat_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
