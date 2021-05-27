# -*- coding:utf-8 -*-
import numpy as np

class Similarity(object):
    def __init__(self):
        pass

    def EuclideanDistance(self, a, b):
        """
        欧氏距离: d^2 = (x1-x2)^2 + (y1-y2)^2
        :param a:
        :param b:
        :return:
        """
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def ManhattanDistance(self, a, b):
        """
        曼哈顿距离:  d = |x1 - x2| + |y1 - y2|
        :param a:
        :param b:
        :return:
        """
        return np.abs(a[0]-b[0]) + np.abs(a[1]-b[1])

    def ChebyshevDistance(self, a, b):
        """
        切比雪夫距离：d = max(|x1-x2|, |y1-y2|)
        :param a:
        :param b:
        :return:
        """
        return np.max(np.abs(a[0]-b[0]), np.abs(a[1]-b[1]))

    def CosineDistance(self, a, b):
        """
        余弦距离
        :param a:
        :param b:
        :return:
        """
        cos = (a[0]*b[0] + a[1]*b[1]) / (np.sqrt(a[0]**2 + a[1]**2) * np.sqrt(b[0]**2 + b[1]**2))
        return cos

    def JaccardSimilarityCoefficient(self, a, b):
        """
        杰卡德相似系数: J = |A交B| / |A并B|
        :param a:
        :param b:
        :return:
        """
        set_a = set(a)
        set_b = set(b)
        dis = float(len(set_a & set_b)) / len(set_a | set_b)
        return dis

    def JaccardSimilarityDistance(self, a, b):
        """
        杰卡德距离: J = (|A并B| - |A交B|) / |A并B|
        :param a:
        :param b:
        :return:
        """
        set_a = set(a)
        set_b = set(b)
        dis = float(len((set_a | set_b) - len(set_a & set_b))) / len(set_a | set_b)
        return dis

