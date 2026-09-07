import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
ratings=pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/ratings.csv")
movies =pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/movies.csv")

ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)
split_index = int(len(ratings_sorted) * 0.8)

train_df = ratings_sorted.iloc[:split_index]
test_df = ratings_sorted.iloc[split_index:]

print("Train:", train_df.shape, "Test:", test_df.shape)
# Split pipe-separated genres into a list
#one-hot encoding 
movies['genre_list'] = movies['genres'].str.split('|')

# One-hot / multi-hot encode genres
all_genres = sorted(set(g for genres in movies['genre_list'] for g in genres))
genre_matrix = pd.DataFrame(0, index=movies['movieId'], columns=all_genres)
#iterrows() goes through the DataFrame one row at a time.
for idx, row in movies.iterrows():
    for genre in row['genre_list']:
        genre_matrix.loc[row['movieId'], genre] = 1

print(genre_matrix.shape)
genre_matrix.head()
movie_similarity = cosine_similarity(genre_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=genre_matrix.index, columns=genre_matrix.index)

movie_similarity_df.shape
#building the users profile 
def build_user_profile(user_id, train_df, genre_matrix, liked_threshold=4.0):
    user_ratings = train_df[(train_df['userId'] == user_id) & (train_df['rating'] >= liked_threshold)]
    liked_movie_ids = user_ratings['movieId']
    
    liked_movie_ids = [m for m in liked_movie_ids if m in genre_matrix.index]
    
    if len(liked_movie_ids) == 0:
        return None  # cold-start case, no liked movies in train
    
    profile = genre_matrix.loc[liked_movie_ids].mean(axis=0)
    return profile
def recommend_for_user_v2(user_id, train_df, genre_matrix, movies, movie_popularity, top_n=10, liked_threshold=4.0):
    profile = build_user_profile(user_id, train_df, genre_matrix, liked_threshold)
    if profile is None:
        return []
    
    sims = cosine_similarity([profile], genre_matrix)[0]
    sim_df = pd.DataFrame({
        'movieId': genre_matrix.index,
        'similarity': sims
    })
    
    # Bring in popularity (rating_count) as a tie-breaker
    sim_df = sim_df.merge(movie_popularity[['movieId', 'rating_count']], on='movieId', how='left')
    sim_df['rating_count'] = sim_df['rating_count'].fillna(0)
    
    # Exclude already-rated movies
    already_rated = train_df[train_df['userId'] == user_id]['movieId'].tolist()
    sim_df = sim_df[~sim_df['movieId'].isin(already_rated)]
    
    # Sort by similarity FIRST, popularity SECOND (as tie-breaker)
    sim_df = sim_df.sort_values(['similarity', 'rating_count'], ascending=[False, False])
    
    return sim_df['movieId'].head(top_n).tolist()
def get_liked_test_movies(user_id, test_df, liked_threshold=4.0):
    user_test = test_df[(test_df['userId'] == user_id) & (test_df['rating'] >= liked_threshold)]
    return set(user_test['movieId'].tolist())

def precision_recall_at_k_for_user(user_id, train_df, test_df, genre_matrix, movies, movie_popularity, k=10, liked_threshold=4.0):
    liked_test_movies = get_liked_test_movies(user_id, test_df, liked_threshold)
    
    if len(liked_test_movies) == 0:
        return None
    
    recommended = recommend_for_user_v2(user_id, train_df, genre_matrix, movies, movie_popularity, top_n=k, liked_threshold=liked_threshold)
    
    if len(recommended) == 0:
        return None
    
    recommended_set = set(recommended)
    hits = recommended_set.intersection(liked_test_movies)
    
    precision = len(hits) / k
    recall = len(hits) / len(liked_test_movies)
    
    return precision, recall
movie_popularity = train_df.groupby('movieId').size().reset_index(name='rating_count')

test_users = test_df['userId'].unique()

precisions = []
recalls = []

for user_id in test_users:
    result = precision_recall_at_k_for_user(user_id, train_df, test_df, genre_matrix, movies, movie_popularity, k=10)
    if result is not None:
        precision, recall = result
        precisions.append(precision)
        recalls.append(recall)

avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)

print(f"Evaluated on {len(precisions)} users (out of {len(test_users)} total test users)")
print(f"Content-Based Filtering v2 — Precision@10: {avg_precision:.4f}")
print(f"Content-Based Filtering v2 — Recall@10: {avg_recall:.4f}")
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# ratings=pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/ratings.csv")
# movies =pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/movies.csv")

# ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)
# split_index = int(len(ratings_sorted) * 0.8)

# train_df = ratings_sorted.iloc[:split_index]
# test_df = ratings_sorted.iloc[split_index:]

# print("Train:", train_df.shape, "Test:", test_df.shape)
# # Split pipe-separated genres into a list
# #one-hot encoding 
# movies['genre_list'] = movies['genres'].str.split('|')

# # One-hot / multi-hot encode genres
# all_genres = sorted(set(g for genres in movies['genre_list'] for g in genres))
# genre_matrix = pd.DataFrame(0, index=movies['movieId'], columns=all_genres)
# #iterrows() goes through the DataFrame one row at a time.
# for idx, row in movies.iterrows():
#     for genre in row['genre_list']:
#         genre_matrix.loc[row['movieId'], genre] = 1

# print(genre_matrix.shape)
# genre_matrix.head()
# movie_similarity = cosine_similarity(genre_matrix)
# movie_similarity_df = pd.DataFrame(movie_similarity, index=genre_matrix.index, columns=genre_matrix.index)

# movie_similarity_df.shape
# #building the users profile 
# def build_user_profile(user_id, train_df, genre_matrix, liked_threshold=4.0):
#     user_ratings = train_df[(train_df['userId'] == user_id) & (train_df['rating'] >= liked_threshold)]
#     liked_movie_ids = user_ratings['movieId']
    
#     liked_movie_ids = [m for m in liked_movie_ids if m in genre_matrix.index]
    
#     if len(liked_movie_ids) == 0:
#         return None  # cold-start case, no liked movies in train
    
#     profile = genre_matrix.loc[liked_movie_ids].mean(axis=0)
#     return profile
# def recommend_for_user_v2(user_id, train_df, genre_matrix, movies, movie_popularity, top_n=10, liked_threshold=4.0):
#     profile = build_user_profile(user_id, train_df, genre_matrix, liked_threshold)
#     if profile is None:
#         return []
    
#     sims = cosine_similarity([profile], genre_matrix)[0]
#     sim_df = pd.DataFrame({
#         'movieId': genre_matrix.index,
#         'similarity': sims
#     })
    
#     # Bring in popularity (rating_count) as a tie-breaker
#     sim_df = sim_df.merge(movie_popularity[['movieId', 'rating_count']], on='movieId', how='left')
#     sim_df['rating_count'] = sim_df['rating_count'].fillna(0)
    
#     # Exclude already-rated movies
#     already_rated = train_df[train_df['userId'] == user_id]['movieId'].tolist()
#     sim_df = sim_df[~sim_df['movieId'].isin(already_rated)]
    
#     # Sort by similarity FIRST, popularity SECOND (as tie-breaker)
#     sim_df = sim_df.sort_values(['similarity', 'rating_count'], ascending=[False, False])
    
#     return sim_df['movieId'].head(top_n).tolist()
# def get_liked_test_movies(user_id, test_df, liked_threshold=4.0):
#     user_test = test_df[(test_df['userId'] == user_id) & (test_df['rating'] >= liked_threshold)]
#     return set(user_test['movieId'].tolist())

# def precision_recall_at_k_for_user(user_id, train_df, test_df, genre_matrix, movies, movie_popularity, k=10, liked_threshold=4.0):
#     liked_test_movies = get_liked_test_movies(user_id, test_df, liked_threshold)
    
#     if len(liked_test_movies) == 0:
#         return None
    
#     recommended = recommend_for_user_v2(user_id, train_df, genre_matrix, movies, movie_popularity, top_n=k, liked_threshold=liked_threshold)
    
#     if len(recommended) == 0:
#         return None
    
#     recommended_set = set(recommended)
#     hits = recommended_set.intersection(liked_test_movies)
    
#     precision = len(hits) / k
#     recall = len(hits) / len(liked_test_movies)
    
#     return precision, recall
# movie_popularity = train_df.groupby('movieId').size().reset_index(name='rating_count')

# test_users = test_df['userId'].unique()

# precisions = []
# recalls = []

# for user_id in test_users:
#     result = precision_recall_at_k_for_user(user_id, train_df, test_df, genre_matrix, movies, movie_popularity, k=10)
#     if result is not None:
#         precision, recall = result
#         precisions.append(precision)
#         recalls.append(recall)

# avg_precision = np.mean(precisions)
# avg_recall = np.mean(recalls)

# print(f"Evaluated on {len(precisions)} users (out of {len(test_users)} total test users)")
# print(f"Content-Based Filtering v2 — Precision@10: {avg_precision:.4f}")
# print(f"Content-Based Filtering v2 — Recall@10: {avg_recall:.4f}")