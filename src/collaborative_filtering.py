import pandas as pd
from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, accuracy
ratings=pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/ratings.csv")
ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)
split_index = int(len(ratings_sorted) * 0.8)
train_df = ratings_sorted.iloc[:split_index]
test_df = ratings_sorted.iloc[split_index:]
print(train_df.shape,test_df.shape)
reader = Reader(rating_scale=(0.5, 5.0))# reader library handles itself everything 
train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
# Pandas DataFrame
# Dataset.load_from_df()
# Surprise Dataset
trainset = train_data.build_full_trainset()
testset = list(zip(test_df['userId'], test_df['movieId'], test_df['rating']))
#in test set we didnt do the same as train bcz we just want to compare the values at last and in train we want to 
#prepare the data for the surprise to use 
print("Trainset size:", trainset.n_ratings)
print("Testset size:", len(testset))
# item-based basic KNN
# Movie A → [ratings from users]
# Movie B → [ratings from users]
#        cosine similarity
#         similarity score
sim_options_item = {
    'name': 'cosine',
    'user_based': False   # item-based
}
algo_item = KNNBasic(sim_options=sim_options_item)
algo_item.fit(trainset)
predictions_item = algo_item.test(testset)
accuracy.rmse(predictions_item)
accuracy.mae(predictions_item)
# used based basic knn
sim_options_user = {
    'name': 'pearson',
    'user_based': True   # user-based
}

algo_user = KNNBasic(sim_options=sim_options_user)
algo_user.fit(trainset)

predictions_user = algo_user.test(testset)

print("User-Based CF (Pearson):")
accuracy.rmse(predictions_user)
accuracy.mae(predictions_user)
sim_options_means = {
    'name': 'cosine',
    'user_based': False
}

algo_means = KNNWithMeans(sim_options=sim_options_means)
algo_means.fit(trainset)

predictions_means = algo_means.test(testset)

print("Item-Based KNNWithMeans:")
accuracy.rmse(predictions_means)
accuracy.mae(predictions_means)
# import pandas as pd
# from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, accuracy
# ratings=pd.read_csv("/Users/harshverma/Downloads/machine learning /movie recommendation system/ml-latest-small/ratings.csv")
# ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)
# split_index = int(len(ratings_sorted) * 0.8)
# train_df = ratings_sorted.iloc[:split_index]
# test_df = ratings_sorted.iloc[split_index:]
# print(train_df.shape,test_df.shape)
# reader = Reader(rating_scale=(0.5, 5.0))# reader library handles itself everything 
# train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
# # Pandas DataFrame
# # Dataset.load_from_df()
# # Surprise Dataset
# trainset = train_data.build_full_trainset()
# testset = list(zip(test_df['userId'], test_df['movieId'], test_df['rating']))
# #in test set we didnt do the same as train bcz we just want to compare the values at last and in train we want to 
# #prepare the data for the surprise to use 
# print("Trainset size:", trainset.n_ratings)
# print("Testset size:", len(testset))
# # item-based basic KNN
# # Movie A → [ratings from users]
# # Movie B → [ratings from users]
# #        cosine similarity
# #         similarity score
# sim_options_item = {
#     'name': 'cosine',
#     'user_based': False   # item-based
# }
# algo_item = KNNBasic(sim_options=sim_options_item)
# algo_item.fit(trainset)
# predictions_item = algo_item.test(testset)
# accuracy.rmse(predictions_item)
# accuracy.mae(predictions_item)
# # used based basic knn
# sim_options_user = {
#     'name': 'pearson',
#     'user_based': True   # user-based
# }

# algo_user = KNNBasic(sim_options=sim_options_user)
# algo_user.fit(trainset)

# predictions_user = algo_user.test(testset)

# print("User-Based CF (Pearson):")
# accuracy.rmse(predictions_user)
# accuracy.mae(predictions_user)
# sim_options_means = {
#     'name': 'cosine',
#     'user_based': False
# }

# algo_means = KNNWithMeans(sim_options=sim_options_means)
# algo_means.fit(trainset)

# predictions_means = algo_means.test(testset)

# print("Item-Based KNNWithMeans:")
# accuracy.rmse(predictions_means)
# accuracy.mae(predictions_means)