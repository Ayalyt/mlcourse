import torch
import torch.nn as nn
import torch.nn.functional as F

class RecommendationModel(nn.Module):
    def __init__(self, uid_max, gender_max, age_max, job_max, embed_dim, movie_id_max, movie_categories_max, movie_title_max,
                 filter_num, window_sizes, sentences_size, dropout_keep, num_classes=5):
     
        super(RecommendationModel, self).__init__()

        # Embedding layers
        self.uid_embed_layer = nn.Embedding(uid_max, embed_dim)
        self.gender_embed_layer = nn.Embedding(gender_max, embed_dim // 2)
        self.age_embed_layer = nn.Embedding(age_max, embed_dim // 2)
        self.job_embed_layer = nn.Embedding(job_max, embed_dim // 2)
        self.movie_id_embed_layer = nn.Embedding(movie_id_max, embed_dim)
        self.movie_categories_embed_layer = nn.Embedding(movie_categories_max, embed_dim)
        self.movie_title_embed_layer = nn.Embedding(movie_title_max, embed_dim)

        # User feature layers
        self.fc_uid = nn.Linear(embed_dim, embed_dim)
        self.fc_gender = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_age = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_job = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_user = nn.Linear(embed_dim * 4, 200)

        self.sentences_size = sentences_size
        self.window_sizes = window_sizes
        
        # Movie feature layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, filter_num, (window_size, embed_dim)) for window_size in window_sizes
        ])
        self.fc_movie = nn.Linear(embed_dim + embed_dim + (len(window_sizes) * filter_num), 200)

        self.dropout = nn.Dropout(dropout_keep)
        self.fc_output = nn.Linear(400, num_classes)

    def forward(self, x):
        uid = x['uid']
        user_gender = x['user_gender']
        user_age = x['user_age']
        user_job = x['user_job']
        movie_id = x['movie_id']
        movie_categories = x['movie_categories']
        movie_titles = x['movie_titles']

        # User features
        uid_embed = self.uid_embed_layer(uid)
        gender_embed = self.gender_embed_layer(user_gender)
        age_embed = self.age_embed_layer(user_age)
        job_embed = self.job_embed_layer(user_job)

        uid_fc = F.relu(self.fc_uid(uid_embed))
        gender_fc = F.relu(self.fc_gender(gender_embed))
        age_fc = F.relu(self.fc_age(age_embed))
        job_fc = F.relu(self.fc_job(job_embed))

        user_feature = torch.cat([uid_fc, gender_fc, age_fc, job_fc], dim=1)
        user_feature = torch.tanh(self.fc_user(user_feature))

        # Movie features
        movie_id_embed = self.movie_id_embed_layer(movie_id)
        movie_categories_embed = self.movie_categories_embed_layer(movie_categories)
        movie_categories_embed = movie_categories_embed.sum(dim=1, keepdim=True).squeeze(1)
        movie_titles_embed = self.movie_title_embed_layer(movie_titles).unsqueeze(1)
        conv_results = [F.relu(conv(movie_titles_embed)).squeeze(3) for conv in self.conv_layers]
        
        maxpool_results = [
            F.max_pool2d(conv.unsqueeze(3), (self.sentences_size - window_size + 1, 1)).squeeze(3).squeeze(2)
            for conv, window_size in zip(conv_results, self.window_sizes)
        ]
        
        pool_output = torch.cat(maxpool_results, dim=1)
        
        movie_feature = torch.cat([movie_id_embed, movie_categories_embed, pool_output], dim=1)
        movie_feature = torch.tanh(self.fc_movie(movie_feature))

        # Combine user and movie features
        combined = torch.cat([user_feature, movie_feature], dim=1)
        combined = self.dropout(combined)
        output = self.fc_output(combined)

        return F.log_softmax(output, dim=1)

if __name__ == "__main__":
    # 初始化模型
    model = ClassificationModel(6040, 2, 7, 21, 32, 3952, 19, 5216, 8, [2, 3, 4, 5], 15, 0.5, num_classes=5)

    # 打印模型结构
    print(model)
