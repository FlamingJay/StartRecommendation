# -*- coding:utf-8 -*-
import numpy as np


class NaiveBayes(object):
    """
    朴素贝叶斯，符合条件独立假设，因而后验概率就变成了  P(yk|x) = P(yk)P(x|y1)P(x|y2)...P(x|yn)
    一般有三种：
    1、多项式模型：特征是离散值，一般会对先验概率和条件概率进行平滑处理，以避免未在训练集中出现特征值x导致后验概率为0的状况发生。
                先验的平滑： P(y_k) = (n_(yk) + a) / (n + ma)其中，n是总样本数，m是总类别数，a是平滑值，n_(yk)是类别为k的样本数，
                若a为1，则是Laplace平滑，0<a<1为Lidstone平滑，a=0不平滑
    2、高斯模型:特征是连续值，假设条件概率符合高斯分布
    3、伯努利模型：特征是离散值，每个特征只能取0或1
    """
    def __init__(self, alpha):
        """
        初始化高斯模型，
        priorProb 存放的是正常/异常的先验概率
        modelParams存放模型参数，即正常/异常高斯模型的各个特征的均值和方差，{label1:{feature1:{mu:1, sigma:2}, feature2：{mu:1, sigma:2}}, label2:{....}}
        :param alpha:平滑系数
        """
        self.priorProb = dict()
        self.modelParams = dict()
        self.alpha = alpha

    def createData(self):
        data = np.array(
            [
                [320, 204, 198, 265],
                [253, 53, 15, 2243],
                [53, 32, 5, 325],
                [63, 50, 42, 98],
                [1302, 523, 202, 5430],
                [32, 22, 5, 143],
                [105, 85, 70, 322],
                [872, 730, 840, 2762],
                [16, 15, 13, 52],
                [92, 70, 21, 693]
            ]
        )
        labels = np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
        return data, labels

    def calMuAndSigma(self, feature):
        mu = np.mean(feature)
        sigma = np.std(feature)
        return(mu, sigma)

    def train(self, data, labels):
        if len(data) != len(labels):
            raise Exception("训练集样本数和标签数不等")
        numData = len(data)
        numFeatures = len(data[0])
        # 先验概率，1为异常，0为正常
        self.priorProb[1] = (
            (sum(labels) + self.alpha) * 1.0 / (numData + self.alpha + len(set(labels)))
        )
        self.priorProb[0] = 1 - self.priorProb[1]

        for c in set(labels):
            self.modelParams[c] = {}
            for i in range(numFeatures):
                feature = data[np.equal(labels, c)][:, i]
                self.modelParams[c][i] = self.calMuAndSigma(feature)

    def gaussian(self, mu, sigma, x):
        return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x-mu)**2 / (2 * sigma**2))

    def predict(self, x):
        """
        预测
        :param x:
        :return:
        """
        label = -1
        maxP = 0
        for key in self.priorProb.keys():
            priorP = self.priorProb[key]
            currentP = 1.0
            feature_p = self.modelParams[key]
            j = 0
            for fp in feature_p.keys():
                currentP *= self.gaussian(feature_p[fp][0], feature_p[fp][1], x[j])
                j += 1
            if currentP * priorP > maxP:
                maxP = currentP * priorP
                label = key

        return label


if __name__=="__main__":
    nb = NaiveBayes(1.0)
    data, labels = nb.createData()
    nb.train(data, labels)
    res = nb.predict(np.array([134, 84, 235, 349]))
    print(res)