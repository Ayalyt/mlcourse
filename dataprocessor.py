import pandas as pd
import numpy as np
import re
import pickle

class Dataprocessor:
    
    def __init__(self):
        self.ratings = []
        self.movies = []
        self.users = []
        self.occupationMapping = dict()
        self.genderMapping = dict()
        self.ageMapping = dict()
        self.genresMapping = dict()
        self.loadData()
        self.ratings_with_movies = self.joinTables()
        self.mapping()
        self.embedding()
        print(self.ratings_with_movies)
        
    
    def loadData(self):
        self.ratings = pd.read_csv('.\\data\\ratings.csv')
        self.movies = pd.read_csv('.\\data\\movies.csv')
        self.users = pd.read_csv('.\\data\\users.csv')
        self.ratings.drop(columns=['Timestamp'],inplace=True)
        self.users.drop(columns=['Area'],inplace=True)
    
    def embedding(self):
        # 将Title中的年份去掉
        pattern = re.compile(r'^(.*)\((\d+)\)$')
        title_map = {val:pattern.match(val).group(1) for ii,val in enumerate(set(self.ratings_with_movies['Title']))}
        self.ratings_with_movies['Title'] = self.ratings_with_movies['Title'].map(title_map)
        
        #电影类型转数字字典
        genres_set = set()
        for val in self.ratings_with_movies['Genres'].str.split('|'):
            genres_set.update(val)
        genres_set.add('<PAD>')
        genres2int = {val:ii for ii, val in enumerate(genres_set)}

        #将电影类型转成等长数字列表，长度是18
        genres_map = {val:[genres2int[row] for row in val.split('|')] for ii,val in enumerate(set(self.ratings_with_movies['Genres']))}
        for key in genres_map:
            for cnt in range(max(genres2int.values()) - len(genres_map[key])):
                genres_map[key].insert(len(genres_map[key]) + cnt,genres2int['<PAD>'])
        self.ratings_with_movies['Genres'] = self.ratings_with_movies['Genres'].map(genres_map)

        #电影Title转数字字典
        title_set = set()
        for val in self.ratings_with_movies['Title'].str.split():
            title_set.update(val)  
        title_set.add('<PAD>')
        title2int = {val:ii for ii, val in enumerate(title_set)}

        #将电影Title转成等长数字列表，长度是15
        title_count = 15
        title_map = {val:[title2int[row] for row in val.split()] for ii,val in enumerate(set(self.ratings_with_movies['Title']))}
        for key in title_map:
            for cnt in range(title_count - len(title_map[key])):
                title_map[key].insert(len(title_map[key]) + cnt,title2int['<PAD>'])
        self.ratings_with_movies['Title'] = self.ratings_with_movies['Title'].map(title_map)
        features, targets = self.ratings_with_movies.drop(['Rating'], axis=1), self.ratings_with_movies['Rating']
        pickle.dump((title_count, title_set, genres2int, features.values, targets.values, self.ratings, self.users, self.movies, self.ratings_with_movies), open('preprocess.p', 'wb'))
    
    def joinTables(self):
        ratings_with_users = pd.merge(self.ratings, self.users, on='UserID')
        ratings_with_movies = pd.merge(ratings_with_users, self.movies, on='MovieID')
        return ratings_with_movies
    
    def mapping(self):
        self.ageMapping = { 
            0: "<18",
            1: "18-24",
            2: "25-34",
            3: "35-44",
            4: "45-49",
            5: "50-55",
            6: "56+"
        }
        self.genderMapping = {
            1: "male",
            0: "female"
        }
        self.occupationMapping = {
            0: "other",
            1: "academic/educator",
            2: "artist",
            3: "clerical/admin",
            4: "college/grad student",
            5: "customer service",
            6: "doctor/health care",
            7: "executive/managerial",
            8: "farmer",
            9: "homemaker",
            10: "K-12 student",
            11: "lawyer",
            12: "programmer",
            13: "retired",
            14: "sales/marketing",
            15: "scientist",
            16: "self-employed",
            17: "technician/engineer",
            18: "tradesman/craftsman",
            19: "unemployed",
            20: "writer"
        }
        self.ratings_with_movies['Gender'] = self.ratings_with_movies['Gender'].replace({v: k for k, v in self.genderMapping.items()}).astype(np.int64)
        self.ratings_with_movies['Occupation'] = self.ratings_with_movies['Occupation'].replace({v: k for k, v in self.occupationMapping.items()}).astype(np.int64)
        self.ratings_with_movies['Age'] = self.ratings_with_movies['Age'].replace({v: k for k, v in self.ageMapping.items()}).astype(np.int64)
    
if __name__ == "__main__":
    dp = Dataprocessor()
    print(dp.ratings_with_movies)
    
    

    
    

    