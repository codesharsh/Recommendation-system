import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ── Rating Prediction Metrics ───────────────────────────────────────────

def rmse(actual, predicted):
    """
    Root Mean Squared Error.
    Penalizes large errors more heavily than small ones.
    """
    return np.sqrt(mean_squared_error(actual, predicted))


def mae(actual, predicted):
    """
    Mean Absolute Error.
    Treats all errors equally; easier to interpret directly
    ("on average, off by X stars").
    """
    return mean_absolute_error(actual, predicted)


def evaluate_predictions(actual, predicted, label="Model"):
    """
    Compute and print RMSE and MAE together for a set of predictions.
    Returns both values as a dict for logging/comparison tables.
    """
    r = rmse(actual, predicted)
    m = mae(actual, predicted)
    print(f"{label} — RMSE: {r:.4f}, MAE: {m:.4f}")
    return {"model": label, "rmse": r, "mae": m}


# ── Ranking Metrics (Precision@K / Recall@K) ────────────────────────────

def get_liked_movies(user_id, df, liked_threshold=4.0):
    """
    Return the set of movieIds a user rated at or above the
    'liked' threshold, within the given dataframe (train or test).
    """
    user_df = df[(df['userId'] == user_id) & (df['rating'] >= liked_threshold)]
    return set(user_df['movieId'].tolist())


def precision_recall_at_k(recommended, liked_movies, k=10):
    """
    Compute Precision@K and Recall@K for a single user, given:
    - recommended: list of recommended movieIds (top-K, in rank order)
    - liked_movies: set of movieIds the user actually liked (ground truth)

    Returns None if there's nothing to evaluate against (no liked
    movies or no recommendations available for this user).
    """
    if len(liked_movies) == 0 or len(recommended) == 0:
        return None

    hits = set(recommended).intersection(liked_movies)

    precision = len(hits) / k
    recall = len(hits) / len(liked_movies)

    return precision, recall


def evaluate_ranking_for_users(user_ids, recommend_fn, test_df, k=10,
                                liked_threshold=4.0, label="Model", verbose=True):
    """
    Compute average Precision@K and Recall@K across a list of users.

    Parameters
    ----------
    user_ids : iterable of user IDs to evaluate
    recommend_fn : callable(user_id) -> list of recommended movieIds
        A pre-configured function (e.g. via functools.partial or a lambda)
        that returns a ranked recommendation list for a given user.
    test_df : pd.DataFrame
        Ground-truth test ratings, used to determine "liked" movies.
    k : int
        Cutoff for Precision@K / Recall@K.
    liked_threshold : float
        Minimum rating to count as "liked".
    label : str
        Name used in the printed summary.

    Returns
    -------
    dict with keys: model, precision_at_k, recall_at_k, n_evaluated, n_total
    """
    precisions, recalls = [], []

    for user_id in user_ids:
        liked = get_liked_movies(user_id, test_df, liked_threshold)
        if len(liked) == 0:
            continue

        recommended = recommend_fn(user_id)
        result = precision_recall_at_k(recommended, liked, k=k)
        if result is None:
            continue

        p, r = result
        precisions.append(p)
        recalls.append(r)

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0

    if verbose:
        print(f"Evaluated on {len(precisions)} users (out of {len(user_ids)} total)")
        print(f"{label} — Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}")

    return {
        "model": label,
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "n_evaluated": len(precisions),
        "n_total": len(user_ids),
    }


# ── Coverage Diagnostic ──────────────────────────────────────────────────

def coverage_report(test_df, known_ids, id_column="movieId", label="Coverage"):
    """
    Report what fraction of test rows involve an ID (user or movie)
    that was NOT present in the training set — i.e. a cold-start case
    the model has no learned information about.
    """
    unseen_mask = ~test_df[id_column].isin(known_ids)
    unseen_count = unseen_mask.sum()
    total = len(test_df)

    print(f"{label}: {unseen_count} / {total} test rows unseen in training "
          f"({unseen_count / total:.2%})")

    return {"unseen_count": unseen_count, "total": total, "unseen_pct": unseen_count / total}


# ── Comparison Table Builder ─────────────────────────────────────────────

def build_comparison_table(results_list):
    """
    Combine a list of result dicts (from evaluate_predictions /
    evaluate_ranking_for_users) into a single pandas DataFrame
    for easy side-by-side comparison across models.
    """
    import pandas as pd
    return pd.DataFrame(results_list)


if __name__ == "__main__":
    # Quick smoke test with dummy data
    actual = [4.0, 3.5, 5.0, 2.0]
    predicted = [3.8, 3.6, 4.5, 2.5]
    evaluate_predictions(actual, predicted, label="Dummy Test")