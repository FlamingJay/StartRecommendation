# -*- coding:utf-8 -*-

class Apriori(object):
    """
    Apriori是挖掘频繁项集和关联规则的算法，频繁项集是经常一起出现的物品集合，采用一种逐层搜索的迭代方法，其中k项集用于探索k+1项集
    1、扫描数据库，累计每个项的计数，并收集满足最小支持度的项，找出频繁1项集的集合，记作L1
    2、使用L1找出频繁2项集的集合L2，使用L2找出L3
    3、如此下去，直至不能再找到频繁k项集，每找出一个Lk需要一次完整的数据库扫描
    --------------------------------------------------------------
    订单编号                  商品名
    --------------------------------------------------------------
    10001                   可乐、面包
    --------------------------------------------------------------
    10002                   面包、纯奶、火腿
    --------------------------------------------------------------
    10003                   面包、纯奶、火腿、泡面
    --------------------------------------------------------------
    10004                   面包、纯奶
    --------------------------------------------------------------
    数字化:  可乐->1，面包->2，纯奶->3，火腿->4，泡面->5
    """
    def __init__(self, minSupport, minConfidence):
         self.minSupport = minSupport
         self.minConfidence = minConfidence
         self.data = self.loadData()

    def loadData(self):
        return [[1,5], [2,3,4], [2,3,4,5], [2,3]]

    def createC1(self, data):
        """
        生成项集1
        :param data:
        :return:
        """
        C1 = list()
        for items in data:
            for item in items:
                if [item] not in C1:
                    C1.append([item])
        return list(map(forzenset, sorted(C1)))

    def scanD(self, Ck):
        """
        扫描数据集，从候选集Ck生成Lk，Lk表示满足最低支持度的元素集合
        :param Ck:
        :return:
        """
        Data = list(map(set, self.data))
        CkCount = {}
        # 计算Ck每个项集在数据集中出现的次数
        for items in Data:
            for one in Ck:
                if one.issubset(items):
                    CkCount.setdefault(one, 0)
                    CkCount[one] += 1
        numItems = len(list(Data))
        Lk = []
        # 获取Ck每个项集的支持度，若大于最小支持度则将数据加入到支持数据Lk中
        supportData = {}
        for key in CkCount:
            support = CkCount[key] * 1.0 / numItems
            if support >= self.minSupport:
                Lk.insert(0, key)
            supportData[key] = support
        return Lk, supportData

    def generateNewCk(self, Lk, k):
        """
        生成下一个候选集
        :param Lk: 频繁项集列表
        :param k: 项集元素个数
        :return:
        """
        nextLk = []
        lenLk = len(Lk)
        # 若两个项集长度为k-1，则必须前k-2项相同才可连接，即求并集，所以[:k-2]的实际作用为取列表的前k-1个元素
        for i in range(lenLk):
            for j in range(i+1, lenLk):
                L1 = list(Lk[i])[:k-2]
                L2 = list(Lk[j])[:k-2]
                if sorted(L1) == sorted(L2):
                    nextLk.append(Lk[i] | Lk[j])
        return nextLk

    def generateLk(self):
        """
        生成频繁项集
        :return:
        """
        C1 = self.createC1(data=self.data)
        L1, supportData = self.scanD(Ck=C1)
        L = [L1]
        k = 2
        while len(L[k-2]) > 0:
            Ck = self.generateNewCk(L[k-2], k)
            Lk, supK = self.scanD(Ck)
            supportData.update(supK)
            L.append(Lk)
            k += 1
        return L, supportData

    def generateRules(self, L, supportData):
        """
        生成规则
        :param L:
        :param supportData:
        :return:
        """
        ruleResult = []
        for i in range(1, len(L)):
            for ck in L[i]:
                Cks = [frozenset([item]) for item in ck]
                self.rulesOfMore(ck, Cks, supportData, ruleResult)
        return ruleResult

    def rulesOfTwo(self, ck, Cks, supportData, ruleResult):
        """
        频繁项只有两个元素
        :param ck:
        :param Cks:
        :param supportData:
        :param ruleResult:
        :return:
        """
        prunedH = []
        for oneCk in Cks:
            conf = supportData[ck] / supportData[ck - oneCk]
            if conf >= self.minConfidence:
                print(ck-oneCk, " --> ", oneCk, " Confidfence is: ", conf)
                ruleResult.append((ck - oneCk, oneCk, conf))
                prunedH.append(oneCk)
        return prunedH

    def rulesOfMore(self, ck, Cks, supportData, ruleResult):
        """
        频繁项中有三个及以上元素的集合，递归生成相关规则
        :param ck:
        :param Cks:
        :param supportData:
        :param ruleResult:
        :return:
        """
        m = len(Cks[0])
        while len(ck) > m:
            Cks = self.rulesOfTwo(ck, Cks, supportData, ruleResult)
            if len(Cks) > 1:
                Cks = self.generateNewCk(Cks, m+1)
                m += 1
            else:
                break


if __name__ == "__main__":
    aprori = Apriori(minSupport=0.5, minConfidence=0.6)
    L, supportData = aprori.generateLk()
    for one in L:
        print("项数为 %s 的频繁项集：" % (L.index(one)+1), one)
    print("supportData:", supportData)
    print("minConf=0.6时:")
    rules = aprori.generateRules(L, supportData)