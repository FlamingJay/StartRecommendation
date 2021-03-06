# -*- coding:utf-8 -*-

import numpy as np
import math

"""
归一化函数测试
"""

class DataNorm:
    def __init__(self):
        self.arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.x_max = max(self.arr)
        self.x_min = min(self.arr)
        self.x_mean = sum(self.arr) / len(self.arr)
        self.x_std = np.std(self.arr) # 标准差

    def Min_Max(self):
        arr_ = list()
        for x in self.arr:
            arr_.append(
                round((x - self.x_min) / (self.x_max - self.x_min), 4)
            )
        print("经过Min_Max标准化后的数据:\n {}".format(arr_))

    def Z_Score(self):
        arr_ = list()
        for x in self.arr:
            arr_.append(
                round((x - self.x_mean) / self.x_std, 4)
            )
        print("经过Z_Score标准化后的数据:\n{}".format(arr_))

    def DecimalScaling(self):
        arr_ = list()
        j = 1
        x_max = max([abs(one) for one in self.arr])
        while x_max / 10 >= 1.0:
            j+=1
            x_max = x_max / 10

        for x in self.arr:
            arr_.append(
                round(x / math.pow(10, j), 4)
            )

        print("经过DecimalScaling后的数据:\n{}".format(arr_))

    def Mean(self):
        arr_ = list()
        for x in self.arr:
            arr_.append(
                round((x - self.x_mean) / (self.x_max - self.x_min), 4)
            )

        print("经过均值标准化后的数据:\n{}".format(arr_))

    def Vector(self):
        arr_ = list()
        for x in self.arr:
            arr_.append(
                round(x / sum(self.arr), 4)
            )

        print("经过向量标准化后的数据:\n{}".format(arr_))

    def exponential(self):
        arr_1 = list()
        for x in self.arr:
            arr_1.append(
                round(math.log10(x) / math.log10(self.x_max), 4)
            )
        print("经过指数转化法log10标准化后的数据:\n{}".format(arr_1))

        arr_2 = list()
        sum_e = sum([math.exp(one) for one in self.arr])
        for x in self.arr:
            arr_2.append(
                round(math.exp(x) / sum_e, 4)
            )
        print("经过指数转化法SoftMax标准化后的数据:\n{}".format(arr_2))

        arr_3 = list()
        for x in self.arr:
            arr_3.append(
                round(1 / (1 + math.exp(-x)), 4)
            )
        print("经过指数转化法Sigmoid标准化后的数据:\n{}".format(arr_3))


if __name__ == "__main__":
    dn = DataNorm()
    dn.Min_Max()
    dn.Z_Score()
    dn.DecimalScaling()
    dn.Mean()
    dn.Vector()
    dn.exponential()