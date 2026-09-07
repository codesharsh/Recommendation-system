import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_loader import load_and_prepare_data


def calculate_movie_statistics(ratings, movies):
    """
    Calculate average rating and rating count for each movie.
    """

    avg_rating = ratings.groupby('movieId')['rating'].mean()
    rating_count = ratings.groupby('movieId')['rating'].count()

    movie_stats = pd.DataFrame({
        'avg_rating': avg_rating,
        'rating_count': rating_count
    }).reset_index()

    movie_stats = movie_stats.merge(
        movies[['movieId', 'title']],
        on='movieId'
    )

    return movie_stats


def calculate_weighted_ratings(movie_stats, quantile=0.91):
    """
    Calculate IMDb-style weighted ratings.
    """

    C = movie_stats['avg_rating'].mean()
    m = movie_stats['rating_count'].quantile(quantile)

    movie_stats = movie_stats.copy()

    movie_stats['weighted_rating'] = (
        (movie_stats['rating_count'] /
         (movie_stats['rating_count'] + m)) * movie_stats['avg_rating']
        +
        (m /
         (movie_stats['rating_count'] + m)) * C
    )

    return movie_stats, C, m


def temporal_split(ratings, train_ratio=0.8):
    """
    Split ratings chronologically to avoid temporal leakage.
    """

    ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)

    split_index = int(len(ratings_sorted) * train_ratio)

    train = ratings_sorted.iloc[:split_index]
    test = ratings_sorted.iloc[split_index:]

    return train, test


def build_baseline_model(train):
    """
    Build the weighted-rating baseline using only training data.
    """

    train_avg_rating = train.groupby('movieId')['rating'].mean()
    train_rating_count = train.groupby('movieId')['rating'].count()

    train_movie_stats = pd.DataFrame({
        'avg_rating': train_avg_rating,
        'rating_count': train_rating_count
    }).reset_index()

    C_train = train['rating'].mean()
    m_train = train_movie_stats['rating_count'].quantile(0.91)

    train_movie_stats['weighted_rating'] = (
        (train_movie_stats['rating_count'] /
         (train_movie_stats['rating_count'] + m_train))
        * train_movie_stats['avg_rating']
        +
        (m_train /
         (train_movie_stats['rating_count'] + m_train))
        * C_train
    )

    movie_weighted_rating_lookup = (
        train_movie_stats
        .set_index('movieId')['weighted_rating']
        .to_dict()
    )

    return movie_weighted_rating_lookup, C_train, m_train, train_movie_stats


def predict_baseline(movie_id, movie_weighted_rating_lookup, C_train):
    """
    Predict a movie rating using the baseline model.

    If the movie was not seen during training,
    use the overall training average as fallback.
    """

    return movie_weighted_rating_lookup.get(movie_id, C_train)


def evaluate_baseline(test, movie_weighted_rating_lookup, C_train):
    """
    Evaluate baseline predictions using RMSE and MAE.
    """

    test = test.copy()

    test['predicted_rating'] = test['movieId'].apply(
        lambda movie_id: predict_baseline(
            movie_id,
            movie_weighted_rating_lookup,
            C_train
        )
    )

    unseen_count = (
        ~test['movieId'].isin(movie_weighted_rating_lookup.keys())
    ).sum()

    rmse = np.sqrt(
        mean_squared_error(
            test['rating'],
            test['predicted_rating']
        )
    )

    mae = mean_absolute_error(
        test['rating'],
        test['predicted_rating']
    )

    return test, unseen_count, rmse, mae


if __name__ == "__main__":

    DATA_PATH = "../data/ml-latest-small"

    ratings, movies, links, tags = load_and_prepare_data(DATA_PATH)

    # ---------------------------------------------------------
    # Overall movie statistics
    # ---------------------------------------------------------

    movie_stats = calculate_movie_statistics(ratings, movies)

    movie_stats.sort_values(
        'avg_rating',
        ascending=False
    ).head(10)

    # ---------------------------------------------------------
    # IMDb-style weighted rating
    # ---------------------------------------------------------

    movie_stats, C, m = calculate_weighted_ratings(movie_stats)

    print("C:", C)
    print("m:", m)

    qualified = movie_stats[
        movie_stats['rating_count'] >= m
    ]

    top_10 = (
        qualified
        .sort_values('weighted_rating', ascending=False)
        .head(10)
    )

    print("\nTop 10 movies by weighted rating:")
    print(
        top_10[
            ['title', 'avg_rating', 'rating_count', 'weighted_rating']
        ]
    )

    # ---------------------------------------------------------
    # Temporal train-test split
    # ---------------------------------------------------------

    train, test = temporal_split(ratings)

    print("\nTrain shape:", train.shape)
    print("Test shape:", test.shape)

    print(
        "Train date range:",
        pd.to_datetime(train['timestamp'], unit='s').min(),
        "to",
        pd.to_datetime(train['timestamp'], unit='s').max()
    )

    print(
        "Test date range:",
        pd.to_datetime(test['timestamp'], unit='s').min(),
        "to",
        pd.to_datetime(test['timestamp'], unit='s').max()
    )

    # ---------------------------------------------------------
    # Build baseline using TRAIN data only
    # ---------------------------------------------------------

    (
        movie_weighted_rating_lookup,
        C_train,
        m_train,
        train_movie_stats
    ) = build_baseline_model(train)

    print("\nC_train:", C_train)
    print("m_train:", m_train)

    # ---------------------------------------------------------
    # Evaluate baseline
    # ---------------------------------------------------------

    (
        test,
        unseen_count,
        rmse,
        mae
    ) = evaluate_baseline(
        test,
        movie_weighted_rating_lookup,
        C_train
    )

    print(
        f"\nTest rows with unseen movies "
        f"(fallback to C): {unseen_count} / "
        f"{len(test)} "
        f"({unseen_count / len(test):.2%})"
    )

    print(f"Baseline RMSE: {rmse:.4f}")
    print(f"Baseline MAE: {mae:.4f}")
# import pandas as pd
# ratings=pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/ratings.csv")
# movies =pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/movies.csv")
# # finding the average rating of the movies 
# avg_rating = ratings.groupby('movieId')['rating'].mean()
# rating_count = ratings.groupby('movieId')['rating'].count()
# movie_stats = pd.DataFrame({
#     'avg_rating': avg_rating,
#     'rating_count': rating_count
# }).reset_index()
# movie_stats = movie_stats.merge(movies[['movieId', 'title']],on='movieId')
# movie_stats.sort_values('avg_rating', ascending=False).head(10)
# #weighted rating formula 
# C = ratings['rating'].mean()
# m = movie_stats['rating_count'].quantile(0.91)

# print("C:", C)
# print("m:", m)
# #imdb formula 
# # weighted rating applying the formula
# # wr= v/(v+m)R + m/(v+m)C
# movie_stats['weighted_rating'] = (
#     (movie_stats['rating_count'] / (movie_stats['rating_count'] + m)) * movie_stats['avg_rating']
#     + (m / (movie_stats['rating_count'] + m)) * C)
# qualified = movie_stats[movie_stats['rating_count'] >= m]
# top_10 = qualified.sort_values('weighted_rating', ascending=False).head(10)
# top_10[['title', 'avg_rating', 'rating_count', 'weighted_rating']]
# # to avoid temporal leak we use timestamp in sorted order  
# ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)
# split_index = int(len(ratings_sorted) * 0.8)
# train = ratings_sorted.iloc[:split_index]
# test = ratings_sorted.iloc[split_index:]
# print("shape of test",train.shape,"shape of train",test.shape)
# # range of timestamp in which train and test works 
# print("Train date range:", pd.to_datetime(train['timestamp'], unit='s').min(), "to", pd.to_datetime(train['timestamp'], unit='s').max())
# print("Test date range:", pd.to_datetime(test['timestamp'], unit='s').min(), "to", pd.to_datetime(test['timestamp'], unit='s').max())
# #weighted rating formula for train set only  
# train_avg_rating = train.groupby('movieId')['rating'].mean()
# train_rating_count = train.groupby('movieId')['rating'].count()

# train_movie_stats = pd.DataFrame({
#     'avg_rating': train_avg_rating,
#     'rating_count': train_rating_count
# }).reset_index()
# C_train = train['rating'].mean()
# m_train = train_movie_stats['rating_count'].quantile(0.91)
# #imdb formula 
# # weighted rating applying the formula
# # wr= v/(v+m)R + m/(v+m)C
# train_movie_stats['weighted_rating'] = (
#     (train_movie_stats['rating_count'] / (train_movie_stats['rating_count'] + m_train)) * train_movie_stats['avg_rating']
#     + (m_train / (train_movie_stats['rating_count'] + m_train)) * C_train
# )
# print("C_train:", C_train)
# print("m_train:", m_train)
# train_movie_stats.head()
# # creating a dictionary 
# movie_weighted_rating_lookup = train_movie_stats.set_index('movieId')['weighted_rating'].to_dict()
# def predict_baseline(movie_id):
#     return movie_weighted_rating_lookup.get(movie_id, C_train)
# test = test.copy()
# test['predicted_rating'] = test['movieId'].apply(predict_baseline)
# unseen_count = (~test['movieId'].isin(movie_weighted_rating_lookup.keys())).sum()
# print(f"Test rows with unseen movies (fallback to C): {unseen_count} / {len(test)} ({unseen_count/len(test):.2%})")
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import numpy as np

# rmse = np.sqrt(mean_squared_error(test['rating'], test['predicted_rating']))
# mae = mean_absolute_error(test['rating'], test['predicted_rating'])

# print(f"Baseline RMSE: {rmse:.4f}")
# print(f"Baseline MAE: {mae:.4f}")