import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ── Tier 1: Collaborative Filtering (SVD) ──────────────────────────────

def recommend_tier1_svd(user_id, svd_model, movies, train_df, top_n=10):
    """
    Recommend movies for a known user using a trained SVD model.
    Used when the user has rating history in the training set.
    """
    all_movie_ids = movies['movieId'].unique()
    already_rated = train_df[train_df['userId'] == user_id]['movieId'].tolist()
    candidates = [m for m in all_movie_ids if m not in already_rated]

    predictions = [(m, svd_model.predict(user_id, m).est) for m in candidates]
    predictions.sort(key=lambda x: x[1], reverse=True)

    return [m for m, score in predictions[:top_n]]


def hybrid_predict_rating(user_id, movie_id, train_df, svd_model, train_movie_stats, C_train):
    """
    Predict a single rating using the hybrid logic:
    - SVD if the user has training history
    - Popularity weighted rating (or global mean) otherwise
    Used for RMSE/MAE evaluation against a test set.
    """
    user_has_history = user_id in train_df['userId'].values

    if user_has_history:
        return svd_model.predict(user_id, movie_id).est

    row = train_movie_stats[train_movie_stats['movieId'] == movie_id]
    if len(row) > 0:
        return row['weighted_rating'].values[0]
    return C_train


# ── Tier 2: Content-Based (new user, stated preferences) ───────────────

def recommend_tier2_content(liked_movie_ids, genre_matrix, movie_popularity, top_n=10):
    """
    Recommend movies for a new user based on a small set of stated
    favorite movies (e.g. an onboarding "pick 5 movies you love" step).
    Used when the user has no rating history but has provided preferences.
    """
    liked_movie_ids = [m for m in liked_movie_ids if m in genre_matrix.index]

    if len(liked_movie_ids) == 0:
        return []

    profile = genre_matrix.loc[liked_movie_ids].mean(axis=0)
    sims = cosine_similarity([profile], genre_matrix)[0]

    sim_df = pd.DataFrame({'movieId': genre_matrix.index, 'similarity': sims})
    sim_df = sim_df.merge(movie_popularity[['movieId', 'rating_count']], on='movieId', how='left')
    sim_df['rating_count'] = sim_df['rating_count'].fillna(0)

    sim_df = sim_df[~sim_df['movieId'].isin(liked_movie_ids)]
    sim_df = sim_df.sort_values(['similarity', 'rating_count'], ascending=[False, False])

    return sim_df['movieId'].head(top_n).tolist()


# ── Tier 3: Popularity Baseline (no information at all) ────────────────

def recommend_tier3_popularity(popularity_ranked, top_n=10):
    """
    Recommend the top-N most popular movies (by weighted rating).
    Used when nothing at all is known about the user.
    """
    return popularity_ranked['movieId'].head(top_n).tolist()


# ── Master switching function ───────────────────────────────────────────

def hybrid_recommend(user_id, train_df, svd_model, genre_matrix, movie_popularity,
                      popularity_ranked, movies, liked_movie_ids=None, top_n=10, verbose=False):
    """
    3-tier hybrid recommendation:
    Tier 1 - SVD, if the user has rating history
    Tier 2 - Content-based, if the user is new but provided liked movies
    Tier 3 - Popularity baseline, if nothing is known about the user
    """
    user_has_history = user_id in train_df['userId'].values

    if user_has_history:
        if verbose:
            print(f"User {user_id}: Tier 1 (SVD) — has training history")
        return recommend_tier1_svd(user_id, svd_model, movies, train_df, top_n)

    elif liked_movie_ids is not None and len(liked_movie_ids) > 0:
        if verbose:
            print(f"User {user_id}: Tier 2 (Content-Based) — new user, provided preferences")
        return recommend_tier2_content(liked_movie_ids, genre_matrix, movie_popularity, top_n)

    else:
        if verbose:
            print(f"User {user_id}: Tier 3 (Popularity) — no history, no preferences given")
        return recommend_tier3_popularity(popularity_ranked, top_n)


# ── Evaluation helper ────────────────────────────────────────────────────

def evaluate_hybrid_rmse(test_df, train_df, svd_model, train_movie_stats, C_train):
    """
    Compute RMSE/MAE for the hybrid's rating-prediction logic
    (Tier 1 + Tier 3 only — Tier 2 cannot be scored against
    historical data, since it requires stated preferences).
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    test_df = test_df.copy()
    test_df['hybrid_predicted'] = test_df.apply(
        lambda row: hybrid_predict_rating(
            row['userId'], row['movieId'], train_df, svd_model, train_movie_stats, C_train
        ),
        axis=1
    )

    rmse = np.sqrt(mean_squared_error(test_df['rating'], test_df['hybrid_predicted']))
    mae = mean_absolute_error(test_df['rating'], test_df['hybrid_predicted'])

    return rmse, mae


if __name__ == "__main__":
    print("hybrid.py — run via a notebook that provides a trained SVD model, "
          "genre_matrix, movie_popularity, and popularity_ranked.")