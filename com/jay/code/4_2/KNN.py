# -*- coding:utf-8 -*-
import numpy as np

class KNN(object):
    def __init__(self, k):
        self.K = k

    def createData(self):
        features = np.array(
            [
                [180, 76], [158, 43], [176, 78], [161, 49]
            ]
        )
        labels = ["M", "F", "M", "F"]
        return features, labels

    def Normalization(self, data):
        """
        数据标准化
        :param data:
        :return:
        """
        maxs = np.max(data, axis=0)
        mins = np.min(data, axis=0)
        new_data = (data - mins) / (maxs - mins)
        return new_data, maxs, mins

    def classify(self, one, data, labels):
        differenceData = data - one
        squareData = np.sum(differenceData ** 2, axis=1)
        sortDistanceIndex = np.argsort(squareData ** 0.5)
        labelCount = dict()
        for i in range(self.K):
            label = labels[sortDistanceIndex[i]]
            labelCount.setdefault(label, 0)
            labelCount[label] += 1

        sortLabelCount = sorted(labelCount.items(), key=lambda x : x[1], reverse=True)
        print(sortLabelCount)
        return sortLabelCount[0][0]


if __name__=="__main__":
    knn = KNN(3)
    features, labels = knn.createData()
    new_data, maxs, mins = knn.Normalization(features)
    one = np.array([175, 76])
    new_one = (one - mins) / (maxs - mins)
    res = knn.classify(new_one, new_data, labels)
    print(res)