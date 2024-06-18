import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import RecommendationModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


with open('test_features.pkl', 'rb') as f:
    test_features = pd.read_pickle(f)
with open('test_labels.pkl', 'rb') as f:
    test_labels = pd.read_pickle(f)
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
# features = np.array(list(zip(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles)), dtype=dtypes)
# # print(features)
# train_features, test_features, train_labels, test_labels = train_test_split(features, targets, test_size=0.2, random_state=42)
batch_size = 256
test_loader = create_dataloader(test_features, test_labels, batch_size)
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
sentences_size = 15

# 模型实例化
model = RecommendationModel(uid_max, gender_max, age_max, job_max, embed_dim, movie_id_max, movie_categories_max, movie_title_max, filter_num, window_sizes, sentences_size, 0.5)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for batch_idx, (uid, gender, age, job, movie_id, categories, titles, labels) in enumerate(test_loader):
            uid, gender, age, job, movie_id, categories, titles, labels = uid.to(device), gender.to(device), age.to(device), job.to(device), movie_id.to(device), categories.to(device), titles.to(device), labels.to(device)
            output = model({'uid': uid, 'user_gender': gender, 'user_age': age, 'user_job': job,
                            'movie_id': movie_id, 'movie_categories': categories, 'movie_titles': titles})
            labels= labels - 1
            test_loss += criterion(output, labels).item()  # 累加 batch 损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大对数概率的索引作为预测类别
            correct += pred.eq(labels.view_as(pred)).sum().item()

            predicted_labels.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

    return np.array(predicted_labels), np.array(true_labels)


predicted_labels, true_labels = test(model, test_loader, criterion)

def test(model, test_loader, criterion):
    model.eval()
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for batch_idx, (uid, gender, age, job, movie_id, categories, titles, labels) in enumerate(test_loader):
            output = model({'uid': uid, 'user_gender': gender, 'user_age': age, 'user_job': job,
                            'movie_id': movie_id, 'movie_categories': categories, 'movie_titles': titles})
            loss = criterion(output, labels.float().unsqueeze(1))
            predicted_labels.extend(output.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return np.array(predicted_labels), np.array(true_labels)

predicted_labels, true_labels = test(model, test_loader, criterion)

print(predicted_labels.round().astype(int), true_label)
print(f"测试集准确率: {accuracy_score(true_labels.round().astype(int), predicted_labels.round().astype(int)):.4f}")

true_labels_binarized = label_binarize(true_labels, classes=[0, 1, 2, 3, 4])
predicted_labels_binarized = label_binarize(predicted_labels.round(), classes=[0, 1, 2, 3, 4])

# 绘制每个类别的P-R曲线
plt.figure()
for i in range(5): 
    precision, recall, _ = precision_recall_curve(true_labels_binarized[:, i], predicted_labels_binarized[:, i])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, marker='.', label=f'Class {i} (area = {pr_auc:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend(loc='best')
plt.show()
