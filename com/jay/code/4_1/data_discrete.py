# -*- coding:utf-8 -*-
import numpy as np
import math

"""
数据离散化方法：根据是否用标签分成监督（4）和非监督（1，2，3）方法
1、等宽分组：确定组数N，确定变量Max和Min，组间隔为(Max-Min)/N
2、等频分组：也称为分位数分组，即分组后每组的变量个数相同。确定组数N，组间隔为变量总数/N
3、单变量分组：也成为秩分组，即对变量进行去重，每个不重复的变量为一组，将相同的变量放进同一组。
4、基于信息熵分组：
"""

class DiscreteByEntropy:
    def __init__(self, group, threshold):
        self.maxGroup = group  # 最大分组数
        self.minInfoThreshold = threshold  # 停止划分的最小熵
        self.result = dict()  # 保存划分结果

    def loadData(self):
        data = np.array(
            [
                [56, 1], [87, 1], [129, 0], [23, 0], [342, 1],
                [641, 1], [63, 0], [2764, 1], [2323, 0], [453, 1],
                [10, 1], [9, 0], [88, 1], [222, 0], [97, 0],
                [2398, 1], [592, 1], [561, 1], [764, 0], [121, 1]
            ]
        )
        return data

    def calEntropy(self, data):
        numData = len(data)
        labelCounts = {}
        for feature in data:
            oneLabel = feature[-1]
            labelCounts.setdefault(oneLabel, 0)
            labelCounts[oneLabel]+=1

        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numData
            shannonEnt -= prob * math.log(prob, 2)
        return shannonEnt

    def split(self, data):
        minEntropy = np.inf
        # 记录最终分割值的下标
        index = -1
        # 按照特征值进行排序
        sortData = data[np.argsort(data[:, 0])]
        # 初始化最终分割数据后的熵
        lastE1, lastE2 = -1, -1

        S1 = dict()
        S2 = dict()

        for i in range(len(sortData)):
            splitData1, splitData2 = sortData[:i+1], sortData[i+1:]
            entropy1, entropy2 = (self.calEntropy(splitData1), self.calEntropy(splitData2))
            entropy = entropy1 * len(splitData1)/len(sortData) + entropy2 * len(splitData2)/len(sortData)
            if entropy < minEntropy:
                index = i
                minEntropy = entropy
                lastE1 = entropy1
                lastE2 = entropy2
        S1["entropy"] = lastE1
        S1["data"] = sortData[:index+1]
        S2["entropy"] = lastE2
        S2["data"] = sortData[index+1:]

        return S1, S2, minEntropy

    def train(self, data):
        """
        对数据进行分组
        :param data:
        :return:
        """
        needSplitKey = [0]
        self.result.setdefault(0, {})
        self.result[0]["entropy"] = np.inf
        self.result[0]["data"] = data
        group = 1
        for key in needSplitKey:
            S1, S2, entropy = self.split(self.result[key]["data"])
            if entropy > self.minInfoThreshold and group < self.maxGroup:
                self.result[key] = S1
                newKey = max(self.result.keys()) + 1
                self.result[newKey] = S2
                needSplitKey.extend([key])
                needSplitKey.extend([newKey])
                group += 1
            else:
                break

if __name__=="__main__":
    dbe = DiscreteByEntropy(group=6, threshold=0.5)
    data = dbe.loadData()
    dbe.train(data)
    print("result {}".format(dbe.result))