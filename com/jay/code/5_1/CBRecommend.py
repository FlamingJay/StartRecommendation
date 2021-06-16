# -*- coding:utf-8 -*-
import json
import pandas as pd
import numpy as np
import math
import random

class CBRecommend(object):
    """
    基于内容的推荐算法Content-Based Recommendation，选用杰卡德相似系数来来衡量两个item之间的相似度，并选取topK个内容进行推荐
    """
    def __init__(self, K):
        self.K = K
        self.item_profile = json.load(open("./data/item_profile.json", "r"))
        self.user_profile = json.load(open("./data/user_profile.json", "r"))

    def get_none_score_item(self, user):
        """
        获取用户未进行评分的item列表
        :param user:
        :return:
        """
        items = pd.read_csv("./data/movies.csv")["MovieID"].values
        data = pd.read_csv("./data/rating.csv")
        have_score_items = data[data["UserID"]==user]["MovieID"].values
        none_score_items = set(items)-set(have_score_items)
        return none_score_items

    def cosUI(self, user, item):
        """
        获取用户对item的喜好程度
        :param user:
        :param item:
        :return:
        """
        Uia = sum(
            np.array(self.user_profile[str(user)]) * np.array(self.item_profile[str(item)])
        )
        Ua = math.sqrt(sum([math.pow(one, 2) for one in self.user_profile[str(user)]]))
        Ia = math.sqrt(sum([math.pow(one, 2) for one in self.item_profile[str(item)]]))

        return Uia / (Ua * Ia)

    def recommend(self, user):
        user_result = {}
        # 获取用户没看过的电影
        item_list = self.get_none_score_item(user)
        # 计算未看电影和用户之间的相似度
        for item in item_list:
            user_result[item] = self.cosUI(user, item)
        if self.K is None:
            result = sorted(user_result.items(), key=lambda k: k[1], reverse=True)
        else:
            result = sorted(user_result.items(), key=lambda k: k[1], reverse=True)[:self.K]

        print(result)

    def evaluate(self):
        evas = []
        data = pd.read_csv("./data/rating.csv")
        for user in random.sample([one for one in range(1, 6041)], 20):
            have_score_items = data[data["UserID"]==user]["MovieID"].values
            items = pd.read_csv("./data/movies.csv")["MovieID"].values
            user_result = {}
            for item in items:
                user_result[item] = self.cosUI(user, item)
            results = sorted(user_result.items(), key=lambda k:k[1], reverse=True)[:len(have_score_items)]
            rec_items = []
            for one in results:
                rec_items.append(one[0])
            eva = len(set(rec_items) & set(have_score_items)) / len(have_score_items)
            evas.append(eva)

        return sum(evas) / len(evas)
if __name__=="__main__":
    cb = CBRecommend(K=10)
    print(cb.evaluate())