# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import random


class KMeans(object):
    """
    步骤：1、初始化K个簇中心；2、计算各个数据到中心的距离；3、按照距离最近原则，将每条数据都划分到最近的簇类中；4、更新簇类中心；5、迭代2-4
    初始化K个簇类中心的方法：
    1、随机选取：指的是映射到2维或3维空间通过肉眼观察取判断
    2、初始聚类：采用层次聚类（Birch、Rock、Canopy）或Canopy算法或者平均质心距离的加权平均值
    """
    def __init__(self, k):
        self.k = k

    def loadData(self, file):
        return pd.read_csv(file, header=0, sep=",")

    def filterAnomalyValue(self, data):
        """
        去除异常值，使用正态分布的99.73%的置信区间，即3σ区间
        :param data:
        :return:
        """
        upper = np.mean(data["price"]) + 3 * np.std(data["price"])
        lower = np.mean(data["price"]) - 3 * np.std(data["price"])
        upper_limit = upper if upper > 5000 else 5000
        lower_limit = lower if lower > 1 else 1
        print("最大异常值为:{}, 最小异常值为：{}".format(upper_limit, lower_limit))

        newData = data[(data["price"] < upper_limit) & (data["price"] > lower_limit)]

        return newData, upper_limit, lower_limit

    def initCenters(self, values, K, Cluster):
        """
        随机初始化簇类中心
        :param values:
        :param K:
        :param Cluster:
        :return:
        """
        random.seed(100)
        oldCenters = list()
        for i in range(K):
            index = random.randint(0, len(values))
            Cluster.setdefault(i, {})
            Cluster[i]["center"] = values[index]
            Cluster[i]["values"] = []

            oldCenters.append(values[index])

        return oldCenters, Cluster

    def distance(self, price1, price2):
        return np.emath.sqrt(pow(price1-price2, 2))

    def kmeans(self, data, maxIters):
        """

        :param data:
        :param K:
        :param maxIters:
        :return:
        """
        Cluster = dict() # 最终聚类结果
        oldCenters, Cluster = self.initCenters(data, self.k, Cluster)
        print("初始聚类中心为：{}".format(oldCenters))
        clusterChanged = True
        i = 0
        while clusterChanged:
            # 计算每个样本距离最近的聚类中心
            for price in data:
                minDistance = np.inf
                minIndex = -1
                for key in Cluster.keys():
                    dis = self.distance(price, Cluster[key]["center"])
                    if dis < minDistance:
                        minDistance = dis
                        minIndex = key
                Cluster[minIndex]["values"].append(price)

            # 计算新的聚类中心
            newCenters = list
            for key in Cluster.keys():
                newCenter = np.mean(Cluster[key]["values"])
                Cluster[key]["center"] = newCenter
                newCenters.append(newCenter)
            print("第{}次迭代后的簇类中心为:{}".format(i, newCenters))
            # 停止条件
            if oldCenters == newCenters or i > maxIters:
                clusterChanged = False
            else:
                oldCenters = newCenters
                i += 1
                for key in Cluster.keys():
                    Cluster[key]["values"] = []

        return Cluster

if __name__ == "__main__":
    file = ""
    km = KMeans(7)
    data = km.loadData(file)
    newData, upper_limit, lower_limit = km.filterAnomalyValue(data)
    Cluster = km.kmeans(newData["price"].values, maxIters=200)
    print(Cluster)