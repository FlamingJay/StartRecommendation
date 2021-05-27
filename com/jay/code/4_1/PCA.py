# -*- coding:utf-8 -*-
import numpy as np
from sklearn import datasets

class PCA(object):
    def __init__(self, k):
        self.k = k

    def __standard(self, data):
        # axis=0 按列取均值
        mean_vector = np.mean(data, axis=0)
        return data - mean_vector

    def __getCovMatrix(self, newData):
        # rowvar=0 表示数据的每一列代表一个feature
        return np.cov(newData, rowvar=0)

    def __getFValueAndFVector(self, covMatrix):
        fValue, fVector = np.linalg.eig(covMatrix)
        return fValue, fVector

    def __getVectorMatrix(self, fValue, fVector):
        fValueSort = np.argsort(fValue)
        return fVector[:, fValueSort[:-(self.k+1):-1]]

    def transform(self, data):
        standard = PCA.__standard(self, data)
        covMatrix = PCA.__getCovMatrix(self, standard)
        print("协方差矩阵为：{}".format(covMatrix))
        fValue, fVector = PCA.__getFValueAndFVector(self, covMatrix)
        print("特征值为：{}".format(fValue))
        print("特征向量为：{}".format(fVector))
        vectorMatrix = PCA.__getVectorMatrix(self, fValue, fVector)
        print("Top{}维向量为：{}".format(self.k, vectorMatrix))
        return np.dot(data, vectorMatrix)

    def loadIris(self):
        data = datasets.load_iris()["data"]
        return data

if __name__=="__main__":
    pcatest = PCA(2)
    data = pcatest.loadIris()
    res = pcatest.transform(data)
    print(res)