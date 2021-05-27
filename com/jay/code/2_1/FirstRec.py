# -*- coding:utf-8 -*-

import json
import os
import random
import math
from Pearaon import Pearson

class FirstRec:

    def __init__(self, file_path, seed, k, n_items, output_path):
        """
        :param file_path: 原始文件路径
        :param seed: 产生随机数的种子
        :param k: 选取的近邻用户个数
        :param n_items: 为每个用户推荐的电影数
        :param output_path: 挑选出的训练和测试数据存储的目录
        """
        self.file_path = file_path
        self.seed = seed
        self.k = k
        self.n_items = n_items
        self.output_path = output_path
        self.users_1000 = self.__select_1000_users()
        self.eva_method = Pearson()
        self.train, self.test = self.__load_and_split_data(self.output_path)

    """
    随机选取1000个用户
    """
    def __select_1000_users(self):
        print("随机选取1000个用户")
        if os.path.exists(self.output_path + "/train.json") and os.path.exists(self.output_path + "/test.json"):
            return list()
        else:
            users = set()
            for file in os.listdir(self.file_path):
                one_path = "{}/{}".format(self.file_path, file)
                print("{}".format(one_path))
                with open(one_path) as fp:
                    for line in fp.readlines():
                        if line.strip().endswith(":"):
                            continue
                        userId, _, _ = line.split(",")
                        users.add(userId)
        # 随机选出1000个
        users_1000 = random.sample(list(users), 1000)
        print(users_1000)

        return users_1000


    """
    加载数据，并拆分成训练集和测试集
    """
    def __load_and_split_data(self, output_path):
        train = dict()
        test = dict()
        if os.path.exists(output_path + "/train.json") and os.path.exists(output_path + "/test.json"):
            print("从文件中加载训练集和测试集")
            train = json.load(open(output_path + "/train.json"))
            test = json.load(open(output_path + "/test.json"))
            print("文件加载完毕")
        else:
            random.seed(self.seed)
            for file in os.listdir(self.file_path):
                one_path = "{}/{}".format(self.file_path, file)
                print("{}".format(one_path))
                with open(one_path, "r") as fp:
                    for line in fp.readlines():
                        line = line.strip("\n")
                        if line.endswith(":"):
                            movieID = line.split(":")[0]
                            continue
                        userId, rate, _ = line.split(",")
                        if userId in self.users_1000:
                            if random.randint(1, 50) == 1:
                                test.setdefault(userId, {})[movieID] = int(rate)
                            else:
                                train.setdefault(userId, {})[movieID] = int(rate)

            print("加载数据到" + output_path)
            with open(output_path + "/train.json",  "w") as json_file:
                json.dump(train, json_file)
            with open(output_path + "/test.json", "w") as json_file:
                json.dump(test, json_file)
            print("加载数据完成")

        return train, test

    def recommend(self, userID):
        """
        为用户userID推荐电影，根据用户相似度来推荐，相似度使用的是皮尔逊系数进行计算
        :param userID:
        :return:
        """
        # 找出最K个相似的用户
        neighborUser = dict()
        for user in self.train.keys():
            if userID != user:
                distance = self.eva_method.pearson(self.train[user], self.train[userID])
                neighborUser[user] = distance

        newNU = sorted(neighborUser.items(), key=lambda k: k[1], reverse=True)

        # 计算出这K个用户所有电影的评分排序，评分由sim*rate(用户相似度*该用户的电影评分)来决定
        movies = dict()
        for (sim_user, sim) in newNU[:self.k]:
            for movieID in self.train[sim_user].keys():
                movies.setdefault(movieID, 0)
                movies[movieID] += sim * self.train[sim_user][movieID]
        newMovies = sorted(movies.items(), key=lambda k: k[1], reverse=True)

        return newMovies

    def evaluate(self, num=30):
        """
        评估推荐系统的准确率
        :param num: 随机抽取num个用户进行测试
        :return:
        """
        print("开始计算准确率")
        precisions = list()
        random.seed(10)
        for userID in random.sample(self.test.keys(), num):
            hit = 0
            result = self.recommend(userID)[:self.n_items]
            for (item, rate) in result:
                if item in self.test[userID]:
                    hit += 1
            precisions.append(hit / self.n_items)
        return sum(precisions) / precisions.__len__()


if __name__ == "__main__":
    file_path = "E:/PythonProject/Recommendation/com/jay/data/netflix-prize-data/combined_data"
    seed = 30
    k = 15
    n_items = 20
    output_path = "E:/PythonProject/Recommendation/com/jay/data/netflix-prize-data/dataset"
    f_rec = FirstRec(file_path, seed, k, n_items, output_path)
    res = f_rec.evaluate()
    print("准确率为:{}".format(res))