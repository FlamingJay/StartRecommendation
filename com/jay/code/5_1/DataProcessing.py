# -*- coding:utf-8 -*-
import pandas as pd
import os
import json


class DataProcessing(object):
    def __init__(self):
        self.genres_all = list()  # 电影类型集合
        self.item_dict = {}  # 电影MovieID对应的电影类型Genres
        self.item_matrix = {}  # 电影特征矩阵（这里的特征指的是电影类型，即MovieID -> Genres）
        self.user_matrix = {}  # 用户特征矩阵（这里的特征指的是对每个电影类型的平均打分，即UserID -> GenresScores）

    def process(self):
        print("开始转换用户数据users.dat...")
        self.process_user_data()
        print("开始转换电影数据movies.data")
        self.process_movie_data()
        print("开始转换用户对电影评分数据...")
        self.process_rating_data()
        print("over")

    def process_user_data(self, file='E:/PythonProject/StartRecommendation/com/jay/data/ml-1m/users.dat'):
        print(os.getcwd())
        fp = pd.read_table(file, sep="::", engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        fp.to_csv('./data/users.csv', index=False)

    def process_movie_data(self, file='E:/PythonProject/StartRecommendation/com/jay/data/ml-1m/movies.dat'):
        fp = pd.read_table(file, sep="::", engine='python', names=['MovieID', 'Title', 'Genres'])
        fp.to_csv('./data/movies.csv', index=False)

    def process_rating_data(self, file='E:/PythonProject/StartRecommendation/com/jay/data/ml-1m/ratings.dat'):
        fp = pd.read_table(file, sep="::", engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        fp.to_csv('./data/rating.csv', index=False)

    def prepare_user_profile(self, file='./data/rating.csv'):
        '''
        计算用户的偏好矩阵
        :param file:
        :return:
        '''
        users = pd.read_csv(file)
        user_ids = set(users["UserID"].values)
        users_rating_dict = {}
        for user in user_ids:
            users_rating_dict.setdefault(str(user), {})

        # 获取用户对每个电影MovieID的评分
        with open(file, 'r') as fr:
            for line in fr.readlines():
                if not line.startswith("UserID"):
                    (user, item, rate) = line.split(",")[:3]
                    users_rating_dict[user][item] = int(rate)

        # 获取用户对每个类型的哪些电影进行了评分
        for user in users_rating_dict.keys():
            print("user is {}".format(user))
            # 用户的平均打分
            score_list = users_rating_dict[user].values()
            avg = sum(score_list) / len(score_list)
            # 遍历每个类型，保证item_profile和user_profile信息矩阵中每列表示的类型一致
            self.user_matrix[user] = []
            for genre in self.genres_all:
                score_all = 0.0
                score_len = 0
                for item in users_rating_dict[user].keys():
                    if genre in self.item_dict[int(item)]:
                        score_all += (users_rating_dict[user][item] - avg)
                        score_len += 1

                if score_len == 0:
                    self.user_matrix[user].append(0)
                else:
                    self.user_matrix[user].append(score_all / score_len)
        json.dump(self.user_matrix, open('./data/user_profile.json', 'w'))
        print("user信息计算完成，保存路径为:{}".format('data/user_profile.json'))

    def prepare_item_profile(self, file='./data/movies.csv'):
        """
        计算电影的特征信息矩阵，计算方式为对电影类型Genres进行编码，比如一个电影MovieID=1是[儿童片|喜剧]，则对应的特征矩阵为[..0,0,1,0,0,1,0,0...]
        :param file:
        :return:
        """
        items = pd.read_csv(file)
        item_ids = set(items["MovieID"].values)
        genres_all = list()
        for item in item_ids:
            genres = items[items["MovieID"]==item]["Genres"].values[0].split("|")
            self.item_dict.setdefault(item, []).extend(genres)
            genres_all.extend(genres)

        self.genres_all = list(set(genres_all))
        for item in self.item_dict.keys():
            self.item_matrix[str(item)] = [0] * len(set(self.genres_all))
            for genre in self.item_dict[item]:
                index = self.genres_all.index(genre)
                self.item_matrix[str(item)][index] = 1
        json.dump(self.item_matrix, open('./data/item_profile.json', 'w'))
        print("item信息计算完成，保存路径为：{}".format('./data/item_profile.json'))

if __name__=='__main__':
    dp = DataProcessing()
    dp.prepare_item_profile()
    dp.prepare_user_profile()