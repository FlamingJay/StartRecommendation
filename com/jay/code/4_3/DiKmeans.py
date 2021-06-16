# -*- coding:utf-8 -*-
from KMeans import KMeans
import numpy as np
import pandas as pd

class DiKmeans(object):
    """
    二分kmeans睡觉哦分层聚类的一种基于kmeans算法实现的
    步骤：
    1、初始化簇类表，使之包含所有数据
    2、对每一个簇类应用k均值聚类算法（k=2）
    3、计算划分后的误差，选择所有被划分的聚簇中总误差（误差平方和SSE）最小的并保存
    4、迭代2和3直到簇类数目达到k后停止
    """
    def __init__(self, k=7):
        self.k = k

    def dikmeans(self, data):
        clusterSSEResult = dict()
        clusterSSEResult.setdefault(0, {})
        clusterSSEResult[0]["values"] = data
        clusterSSEResult[0]["sse"] = np.inf
        clusterSSEResult[0]["center"] = np.mean(data)

        while len(clusterSSEResult) < self.k:
            maxSSE = -np.inf
            maxSSEKey = 0
            # 找到最大SSE对应数据，进行kmeans聚类
            for key in clusterSSEResult.keys():
                if clusterSSEResult[key]["sse"] > maxSSE:
                    maxSSE = clusterSSEResult[key]["sse"]
                    maxSSEKey = key
            # 对选出SSE最大的数据进行二次聚类，并把之前的聚簇给删掉
            clusterResult = KMeans(2).kmeans(clusterSSEResult[key]["values"], maxIters=200)
            del clusterSSEResult[maxSSEKey]

            # 在原位置上添加新聚簇中的第一个
            clusterSSEResult.setdefault(maxSSEKey, {})
            clusterSSEResult[maxSSEKey]["center"] = clusterSSEResult[0]["center"]
            clusterSSEResult[maxSSEKey]["values"] = clusterSSEResult[0]["values"]
            clusterSSEResult[maxSSEKey]["sse"] = self.SSE(clusterResult[0]["values"], clusterResult[0]["center"])
            # 在末尾添加新聚簇中的第二个
            maxKey = max(clusterSSEResult.keys()) + 1
            clusterSSEResult.setdefault(maxKey, {})
            clusterSSEResult[maxKey]["center"] = clusterResult[1]["center"]
            clusterSSEResult[maxKey]["values"] = clusterResult[1]["values"]
            clusterSSEResult[maxKey]["sse"] = self.SSE(clusterResult[1]["values"], clusterResult[1]["center"])

        return clusterSSEResult

    def loadData(self, file):
        return pd.read_csv(file, header=0, sep=",")

    def filterAnomalyValue(self, data):
        upper = np.mean(data["price"]) + 3 * np.std(data["price"])
        lower = np.mean(data["price"]) - 3 * np.std(data["price"])
        upper_limit = upper if upper > 5000 else 5000
        lower_limit = lower if lower > 1 else 1
        print("最大异常值为:{}, 最小异常值为：{}".format(upper_limit, lower_limit))

        newData = data[(data["price"] < upper_limit) & (data["price"] > lower_limit)]

        return newData, upper_limit, lower_limit

    def SSE(self, data, mean):
        newData = np.mat(data) - mean
        return (newData * newData.T).tolist()[0][0]

if __name__ == "__main__":
    file = ""
    km = DiKmeans(7)
    data = km.loadData(file)
    newData, upper_limit, lower_limit = km.filterAnomalyValue(data)
    clusterSSE = km.dikmeans(newData["price"].values)
    print(clusterSSE)