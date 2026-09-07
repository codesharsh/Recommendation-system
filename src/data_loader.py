import pandas as pd


def load_data(data_path):
    """
    Load MovieLens datasets from the specified directory.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the MovieLens CSV files.

    Returns
    -------
    ratings : pd.DataFrame
        User ratings data.
    movies : pd.DataFrame
        Movie information and genres.
    links : pd.DataFrame
        MovieLens-to-external-ID mappings.
    tags : pd.DataFrame
        User-generated movie tags.
    """

    ratings = pd.read_csv(f"{data_path}/ratings.csv")
    movies = pd.read_csv(f"{data_path}/movies.csv")
    links = pd.read_csv(f"{data_path}/links.csv")
    tags = pd.read_csv(f"{data_path}/tags.csv")

    return ratings, movies, links, tags


def prepare_movies(movies):
    """
    Prepare movie genre information.

    Adds:
    - genre_list: list of genres for each movie
    - num_genres: number of genres for each movie
    """

    movies = movies.copy()

    movies['genre_list'] = movies['genres'].str.split('|')
    movies['num_genres'] = movies['genre_list'].apply(len)

    return movies


def prepare_ratings(ratings):
    """
    Convert Unix timestamps into datetime values.
    """

    ratings = ratings.copy()

    ratings['datetime'] = pd.to_datetime(
        ratings['timestamp'],
        unit='s'
    )

    return ratings


def load_and_prepare_data(data_path):
    """
    Load and prepare all MovieLens datasets.

    Returns
    -------
    ratings : pd.DataFrame
    movies : pd.DataFrame
    links : pd.DataFrame
    tags : pd.DataFrame
    """

    ratings, movies, links, tags = load_data(data_path)

    movies = prepare_movies(movies)
    ratings = prepare_ratings(ratings)

    return ratings, movies, links, tags


if __name__ == "__main__":

    DATA_PATH = "ml-latest-small"

    ratings, movies, links, tags = load_and_prepare_data(DATA_PATH)

    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    print("Links shape:", links.shape)
    print("Tags shape:", tags.shape)

    print("\nRatings:")
    print(ratings.head())

    print("\nMovies:")
    print(movies.head())
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import Counter
# ratings=pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/ratings.csv")
# movies =pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/movies.csv")
# links  =pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/links.csv")
# tags   =pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/tags.csv")
# print(ratings.shape)
# print(movies.shape)
# print(tags.shape)
# print(links.shape)
# # check for any null values ,,understand the data 
# print(ratings.info())
# print(movies.info())
# print(movies.duplicated().sum())
# print(ratings.duplicated().sum())
# #no null values and no duplicates 
# print("ratingsscale, minimum=",ratings['rating'].min(),"maximum=",ratings['rating'].max(),"unique values")
# print("unique ratings",sorted(ratings['rating'].unique()))
# ratings['rating'].value_counts().sort_index().plot(
#     kind='bar',
#     title='Rating Distribution'
# )
# # how many times this rating is recieved 
# #ratings per user 
# ratings_per_user=ratings.groupby('userId').size()
# print("min_rating",ratings_per_user.min(),
#       "max_rating",ratings_per_user.max(),
#       "rating_per_",ratings_per_user.mean())
# ratings_per_movie=ratings.groupby('movieId').size()
# print("min_rating",ratings_per_movie.min(),
#       "max_rating",ratings_per_movie.max(),
#       "rating_per_",ratings_per_movie.mean())
# ratings_per_movie.sort_values(ascending=False).reset_index(drop=True).plot(
#     title='Ratings per Movie (sorted, long-tail)'
# )
# plt.xlabel('Movie rank')
# plt.ylabel('Number of ratings')
# plt.show()
# #sparsity check means (no of enteries/total no of possible enteries) 
# num_users = ratings['userId'].nunique()
# num_movies = ratings['movieId'].nunique()
# num_ratings = len(ratings)
# print("users",num_users,"movies",num_movies,"ratings",num_ratings)
# print(f"sparsity: {1 - (num_ratings / (num_users * num_movies)):.4%}")
# # genre exploration
# movies['genre_list'] = movies['genres'].str.split('|')
# all_genres = [g for genre_list in movies['genre_list'] for g in genre_list]
# genre_counts = Counter(all_genres)
# #creting a new dataset to see the freq of generes
# #items make pairs 
# genre_df = pd.DataFrame(
#     genre_counts.items(),
#     columns=['genre', 'count']
# ).sort_values('count', ascending=False)
# # ploting 
# genre_df.plot(kind='bar', x='genre', y='count', title='Genre Frequency', legend=False)
# plt.xticks(rotation=75)
# plt.show()
# movies['num_genres'] = movies['genre_list'].apply(len)
# print("single genre",(movies['num_genres'] == 1).sum())
# print("multiple genre",(movies['num_genres'] > 1).sum())
# #timestamp conversion
# ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
# print("Date range:", ratings['datetime'].min(), "to", ratings['datetime'].max())
# #merging the ratings and movies 
# merged = ratings.merge(movies, on='movieId', how='left')
# print("Ratings rows:", len(ratings))
# print("Merged rows:", len(merged))
# print("Unmatched:", merged['title'].isnull().sum())
