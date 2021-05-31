# -*- coding:utf-8 -*-
import numpy as np
import operator
import math

class DecisionTree(object):
    def __init__(self):
        self.tree = dict

    def loadData(self):
        data = [
            [2, 2, 1, 0, "yes"],
            [2, 2, 1, 1, "no"],
            [1, 2, 1, 0, "yes"],
            [0, 0, 0, 0, "yes"],
            [0, 0, 0, 1, "no"],
            [1, 0, 0, 1, "yes"],
            [2, 1, 1, 0, "no"],
            [2, 0, 0, 0, "yes"],
            [0, 1, 0, 0, "yes"],
            [2, 1, 0, 1, "yes"],
            [1, 2, 0, 0, "no"],
            [0, 1, 1, 1, "no"]
        ]
        features = ["天气", "温度", "湿度", "风速"]
        return data, features

    def ShannonEnt(self, data):
        """
        计算初始的香农熵，也就是label的熵
        :param data:
        :return:
        """
        numData = len(data)
        labelCounts = dict()
        for feature in data:
            oneLabel = feature[-1]
            labelCounts.setdefault(oneLabel, 0)
            labelCounts[oneLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numData
            shannonEnt -= prob * math.log2(prob)

        return shannonEnt

    def splitData(self, data, axis, value):
        """
        划分数据集，将相同特征的数据集(不再包含axis这个特征)抽取出来
        :param data:待划分的数据集
        :param axis: 特征
        :param value: 特征返回值
        :return:
        """
        retData = []
        for feature in data:
            if feature[axis] == value:
                reduceFeature = feature[:axis]
                reduceFeature.extend(feature[axis+1:])
                retData.append(reduceFeature)

        return retData

    def chooseBestFeatureToSplit(self, data):
        """
        选择当前最优的特征进行数据集划分
        :param data:
        :return:
        """
        numFeature = len(data[0]) - 1
        baseEntropy = self.ShannonEnt(data)
        bestInfoGain = 0.0
        bestFeature = -1

        for i in range(numFeature):
            featureList = [result[i] for result in data]
            uniqueFeatureList = set(featureList)
            newEntropy = 0.0
            # 计算这个特征的信息熵
            for value in uniqueFeatureList:
                splitDataset = self.splitData(data, i, value)
                prob = len(splitDataset) / float(len(data))
                newEntropy += prob * self.ShannonEnt(splitDataset)
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestFeature = i
                bestInfoGain = infoGain
        return bestFeature

    def majorityCnt(self, labelsList):
        """
        投票
        :param labelsList:
        :return:
        """
        labelCount = dict()
        for vote in labelsList:
            if vote not in labelCount.keys():
                labelCount[vote] = 0
            labelCount[vote] += 1
        sortedLabelCount = sorted(labelCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedLabelCount[0][0]

    def createTree(self, data, features):
        features = list(features)
        labelList = [line[-1] for line in data]
        # 若类别相同
        if labelList.count(labelList[0]) == len(labelList):
            return labelList[0]
        # 若只有一个特征
        if len(data[0]) == 1:
            return self.majorityCnt(labelList)
        bestFeature = self.chooseBestFeatureToSplit(data)
        bestFeatureName = features[bestFeature]
        myTree = {bestFeatureName: {}}
        # 清空features[bestFeature]，在下一次使用时清零
        del (features[bestFeature])
        featureValues = [example[bestFeature] for example in data]
        uniqueFeatureValues = set(featureValues)
        for value in uniqueFeatureValues:
            subFeatures = features[:]
            # 递归调用创建决策树函数
            myTree[bestFeatureName][value] = self.createTree(self.splitData(data, bestFeature, value), subFeatures)

        self.tree = myTree
        return myTree

    def predict(self, tree:dict, features:list, x):
        """
        预测
        :param tree:
        :param features:
        :param x:
        :return:
        """
        for key1 in tree.keys():
            secondDict:dict = tree[key1]
            featureIndex = features.index(key1)
            for key2 in secondDict.keys():
                if x[featureIndex] == key2:
                    if type(secondDict[key2]).__name__ == "dict":
                        classLabel = self.predict(secondDict[key2], features, x)
                    else:
                        classLabel = secondDict[key2]

        return classLabel



if __name__ == "__main__":
    dtree = DecisionTree()
    data, features = dtree.loadData()
    myTree = dtree.createTree(data, features)
    # label = dtree.predict(myTree, features, [1, 1, 1, 0])
    # print(label)