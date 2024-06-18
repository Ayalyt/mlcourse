import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import RecommendationModel
from sklearn.model_selection import train_test_split
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # 加载预处理数据
with open('preprocess.p', 'rb') as f:
    title_count, title_set, genres2int, features, targets, ratings, users, movies, ratings_with_movies = pd.read_pickle(f)
targets = np.array(targets, dtype=np.int64)
def create_dataloader(features, labels, batch_size):
    uid = np.array([x[0] for x in features], dtype=np.int64)
    user_gender  = np.array([x[1] for x in features], dtype=np.int64)
    user_age  = np.array([x[2] for x in features], dtype=np.int64)
    user_job = np.array([x[3] for x in features], dtype=np.int64)
    movie_id  = np.array([x[4] for x in features], dtype=np.int64)

    movie_categories = np.array([f[5] for f in features], dtype=np.int64)
    movie_titles = np.array([f[6] for f in features], dtype=np.int64)
    movie_categories = [np.array(x, dtype=np.int64) for x in movie_categories.tolist()]
    movie_titles = [np.array(x, dtype=np.int64) for x in movie_titles.tolist()]
    # print(uid, user_gender, user_age)

    dataset = TensorDataset(
        torch.tensor(uid, dtype=torch.int64),
        torch.tensor(user_gender, dtype=torch.int64),
        torch.tensor(user_age, dtype=torch.int64),
        torch.tensor(user_job, dtype=torch.int64),
        torch.tensor(movie_id, dtype=torch.int64),
        torch.tensor(np.vstack(movie_categories), dtype=torch.int64),
        torch.tensor(np.vstack(movie_titles), dtype=torch.int64),
        torch.tensor(labels, dtype=torch.int64)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # 数据准备
# uid = np.array([x[0] for x in features], dtype=np.int64)
# movie_id = np.array([x[1] for x in features], dtype=np.int64)
# user_gender  = np.array([x[2] for x in features], dtype=np.int64)
# user_age = np.array([x[3] for x in features], dtype=np.int64)
# user_job = np.array([x[4] for x in features], dtype=np.int64)
# movie_titles  = [np.array(x[5], dtype=np.int64) for x in features]
# movie_categories = [np.array(x[6], dtype=np.int64) for x in features]
   
# dtypes = [
#     ('UserID', np.int64),
#     ('Gender', np.int64), 
#     ('Age', np.int64),
#     ('Occupation', np.int64),    
#     ('MovieID', np.int64),
#     ('Genres', 'O'),  
#     ('Titles', 'O')    
# ]

# # 创建结构化数组
# features = np.array(list(zip(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles)), dtype=dtypes)
# # print(features)
# train_features, test_features, train_labels, test_labels = train_test_split(features, targets, test_size=0.2, random_state=42)

# import pickle

# with open('train_features.pkl', 'wb') as f:
#     pickle.dump(train_features, f)

# with open('test_features.pkl', 'wb') as f:
#     pickle.dump(test_features, f)

# with open('train_labels.pkl', 'wb') as f:
#     pickle.dump(train_labels, f)

# with open('test_labels.pkl', 'wb') as f:
#     pickle.dump(test_labels, f)
with open('train_features.pkl', 'rb') as f:
    train_features = pd.read_pickle(f)
with open('train_labels.pkl', 'rb') as f:
    train_labels = pd.read_pickle(f)

batch_size = 256
train_loader = create_dataloader(train_features, train_labels, batch_size)
# print(users.dtypes)

uid_max = users['UserID'].max() + 1
gender_max = users['Gender'].nunique()
age_max = users['Age'].nunique()
job_max = users['Occupation'].nunique()
embed_dim = 32
movie_id_max = movies['MovieID'].max() + 1
movie_categories_max = len(genres2int)
movie_title_max = len(title_set)
filter_num = 8
window_sizes = {2, 3, 4, 5}
sentences_size = 15 # = 15
print(uid_max, gender_max, age_max, job_max)

# 模型实例化
model = RecommendationModel(uid_max, gender_max, age_max, job_max, embed_dim, movie_id_max, movie_categories_max, movie_title_max, filter_num, window_sizes, sentences_size, 0.5)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()
model.to(device)

# 训练函数
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (uid, gender, age, job, movie_id, categories, titles, labels) in enumerate(train_loader):
        uid, gender, age, job, movie_id, categories, titles, labels = uid.to(device), gender.to(device), age.to(device), job.to(device), movie_id.to(device), categories.to(device), titles.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model({
            'uid': uid, 
            'user_gender': gender, 
            'user_age': age, 
            'user_job': job,
            'movie_id': movie_id, 
            'movie_categories': categories, 
            'movie_titles': titles
        })
        labels = labels - 1 
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(uid)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

num_epochs = 5
for epoch in range(1, num_epochs + 1):
    train(model, train_loader, optimizer, criterion, epoch)

# 保存模型
torch.save(model.state_dict(), "recommendation_model.pth")
