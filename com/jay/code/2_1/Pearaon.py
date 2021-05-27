import math

class Pearson():
    def __init__(self):
        pass

    def pearson(self, rating1:dict, rating2:dict):
        """
        计算Pearson相关系数
        :param rating1: 用户1的评分系数，{"moive1": rate1, "movie2": rate2,...,"movieN": rateN}
        :param rating2: 用户2的评分系数，{"moive1": rate1, "movie2": rate2,...,"movieN": rateN}
        :return:
        """
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        num = 0
        for key in rating1.keys():
            if key in rating2.keys():
                num += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += math.pow(x, 2)
                sum_y2 += math.pow(y, 2)
        if num == 0:
            return 0
        denominator = math.sqrt(sum_x2 - math.pow(sum_x, 2) / num) * math.sqrt(sum_y2 - math.pow(sum_y, 2) / num)
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y)/num) / denominator


if __name__ == "__main__":
    dict1 = {"nba": 10, "cba": 9, "wwe":3}
    dict2 = {"wwe": 9, "nba":4, "score":17}
    dict3 = {"nba": 9, "cba": 6, "wwe": 1}
    res1 = Pearson().pearson(dict1, dict2)
    res2 = Pearson().pearson(dict1, dict3)
    print("res1:{}, res2:{}".format(res1, res2))