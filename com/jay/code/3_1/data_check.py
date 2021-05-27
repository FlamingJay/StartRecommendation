import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def getRatings(file_path):
    rates = pd.read_table(file_path, header=None,sep="::", names=["userID", "movieID", "rate", "timestamp"])
    print("userID的范围为:<{}, {}>"
          .format(rates["userID"].values.min, rates["userID"].values.max))
    print("movieID的范围为:<{}, {}>"
          .format(min(rates["movieID"].values), max(rates["movieID"].values)))
    print("rate的范围为:<{}, {}>"
          .format(min(rates["rate"].values), max(rates["rate"].values)))
    print("数据总条数为:\n{}".format(
        rates.count()
    ))
    print("数据前5条记录为:\n{}".format(rates.head(5)))
    df = rates["userID"].groupby(rates["userID"])
    print("用户评分记录最少条数为:{}".format(df.count().min()))

    scores = rates["rate"].groupby(rates["rate"]).count()
    for x, y in zip(scores.keys(), scores.values):
        plt.text(x, y+2, "%.0f" % y, ha="center", va="bottom", fontsize=12)
    plt.bar(scores.keys(), scores.values, fc="r", tick_label=scores.keys())
    plt.xlabel("评分分数")
    plt.ylabel("对应人数")
    plt.title("评分分数对应人数统计")
    plt.show()


def getMovies(file_path):
    movie = pd.read_table(file_path, header=None, sep="::", names=["movieID", "title", "genres"])
    print("movies总条数:{}".format(movie.count()))

    moviesDict = dict()
    for line in movie["genres"]:
        for one in line.split("|"):
            moviesDict.setdefault(one, 0)
            moviesDict[one] += 1
    print("电影类型总数为:{}".format(len(moviesDict)))
    print("电影类型分别是：{}".format(moviesDict.keys()))
    print(moviesDict)

    newMD = sorted(moviesDict.items(), key= lambda x: x[1], reverse=True)
    labels = [newMD[i][0] for i in range(len(newMD))]
    values = [newMD[i][1] for i in range(len(newMD))]
    explode = [x * 0.01 for x in range(len(newMD))]
    plt.axes(aspect=1)
    plt.pie(x=values, labels=labels, explode=explode,
            autopct="%3.1f %%", shadow=False, labeldistance=1.1, startangle=0, pctdistance=0.8, center=(-1, 0))
    plt.legend(loc=7, bbox_to_anchor=(1.3, 1.0), ncol=3, fancybox=True, shadow=True, fontsize=6)
    plt.show()


def getUsers(file_path):
    users = pd.read_table(file_path, header=None, sep="::", names=["userID", "gender", "age", "Occupation", "zip-code"])
    print("userID的范围是:<{}, {}>".format(min(users["userID"].values), max(users["userID"].values)))
    print("数据总条数为：{}".format(users.count()))

    userGender = users["gender"].groupby(users["gender"]).count()
    print(userGender)

    plt.axes(aspect=1)
    plt.pie(x=userGender.values, labels=userGender.keys(), autopct="%3.1f %%")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()

if __name__=="__main__":
    # getRatings("E:/PythonProject/Recommendation/com/jay/data/ml-1m/ratings.dat")
    # getMovies("E:/PythonProject/Recommendation/com/jay/data/ml-1m/movies.dat")
    getUsers("/com/jay/data/ml-1m/users.dat")