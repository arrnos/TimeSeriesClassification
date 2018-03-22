import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA, PCA, IncrementalPCA, FactorAnalysis, RandomizedPCA, FastICA

# np.set_printoptions(threshold=np.inf)
# pd.set_option('display.max_rows', None)

from src.lunwen_fea_abstract import *
from src.lunwen_model import *


# GBDT特征选择和排序
def fea_select(X_df, y):
    from sklearn.ensemble import GradientBoostingClassifier
    gbdt = GradientBoostingClassifier(
        init=None,
        learning_rate=0.1,
        loss='deviance',
        max_depth=3,
        max_features=None,
        max_leaf_nodes=None,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=100,
        random_state=None,
        subsample=1.0,
        verbose=0,
        warm_start=False)

    gbdt.fit(X_df, y)

    score = gbdt.feature_importances_
    score = [float('%.3f' % (x)) for x in score]
    score_series = pd.Series(score, index=X_df.columns)
    score_series.sort_values(ascending=False, inplace=True)
    cumsum_score = np.cumsum(score_series)
    score_df = pd.concat([score_series, cumsum_score], axis=1)
    score_df.columns = ['score', 'cum_score']
    score = (score_df[score_df['cum_score'] <= 0.70]['score']).sort_values(ascending=True)
    print(score)
    score.plot(kind='barh', use_index=True)
    plt.xlabel("feature_importance / %")
    plt.ylabel("feature")
    print(score.shape)
    plt.show()


#  Relief距离度量
def distanceNorm(Norm, D_value):
    # Norm for distance
    if Norm == '1':
        counter = np.absolute(D_value)
        counter = np.sum(counter)
    elif Norm == '2':
        counter = np.power(D_value, 2)
        counter = np.sum(counter)
        counter = np.sqrt(counter)
    elif Norm == 'Infinity':
        counter = np.absolute(D_value)
        counter = np.max(counter)
    else:
        raise Exception('We will program this later......')

    return counter


# Relief特征选择核心算法
def fit(features, labels, iter_ratio, k, norm):
    # initialization
    (n_samples, n_features) = np.shape(features)
    distance = np.zeros((n_samples, n_samples))
    weight = np.zeros(n_features)
    labels = list(map(int, labels))

    # compute distance
    for index_i in range(n_samples):
        for index_j in range(index_i + 1, n_samples):
            D_value = features[index_i] - features[index_j]
            distance[index_i, index_j] = distanceNorm(norm, D_value)
    distance += distance.T

    # start iteration
    for iter_num in range(int(iter_ratio * n_samples)):
        # random extract a sample
        index_i = randrange(0, n_samples, 1)
        self_features = features[index_i]

        # initialization
        nearHit = list()
        nearMiss = dict()
        n_labels = list(set(labels))
        termination = np.zeros(len(n_labels))
        del n_labels[n_labels.index(labels[index_i])]
        for label in n_labels:
            nearMiss[label] = list()
        distance_sort = list()

        # search for nearHit and nearMiss
        distance[index_i, index_i] = np.max(distance[index_i])  # filter self-distance
        for index in range(n_samples):
            distance_sort.append([distance[index_i, index], index, labels[index]])

        distance_sort.sort(key=lambda x: x[0])

        for index in range(n_samples):
            # search nearHit
            if distance_sort[index][2] == labels[index_i]:
                if len(nearHit) < k:
                    nearHit.append(features[distance_sort[index][1]])
                else:
                    termination[distance_sort[index][2]] = 1
                    # search nearMiss
            elif distance_sort[index][2] != labels[index_i]:
                if len(nearMiss[distance_sort[index][2]]) < k:
                    nearMiss[distance_sort[index][2]].append(features[distance_sort[index][1]])
                else:
                    termination[distance_sort[index][2]] = 1

            if list(map(int, list(termination))).count(0) == 0:
                break

                # update weight
        nearHit_term = np.zeros(n_features)
        for x in nearHit:
            nearHit += np.abs(np.power(self_features - x, 2))
        nearMiss_term = np.zeros((len(list(set(labels))), n_features))
        for index, label in enumerate(nearMiss.keys()):
            for x in nearMiss[label]:
                nearMiss_term[index] += np.abs(np.power(self_features - x, 2))
            weight += nearMiss_term[index] / (k * len(nearMiss.keys()))
        weight -= nearHit_term / k


        # print weight/(iter_ratio*n_samples)
    return weight / (iter_ratio * n_samples)


# ReliefF算法调用
def ReliefF(X_df, y, iter_ratio, k, norm, score_ReliefF=0.01, is_print=True, is_show=False):
    features = X_df.values
    labels = y
    # n次计算求平均值
    n = 10
    weight = np.zeros(len(X_df.columns)).tolist()
    for x in range(n):
        weight_i = fit(features=features, labels=labels, iter_ratio=iter_ratio, k=k, norm=norm)
        weight = map(lambda x, y: x + y, weight, weight_i)
    weight = list(map(lambda x: x / n, weight))

    # 排序筛选敏感特征
    sum_weight = np.sum(weight)
    score = [x / sum_weight for x in weight]
    score = [float('%.3f' % (x)) for x in score]

    # print("原始分数：\n",score)

    score_series = pd.Series(score, index=X_df.columns)
    score_series.sort_values(ascending=False, inplace=True)
    cumsum_score = np.cumsum(score_series)
    score_df = pd.concat([score_series, cumsum_score], axis=1)
    score_df.columns = ['score', 'cum_score']
    # score = (score_df[score_df['cum_score'] <= 0.70]['score']).sort_values(ascending=True)
    score = (score_df[score_df['score'] >= score_ReliefF]['score']).sort_values(ascending=True)

    # print("去掉不重要特征：\n",score)
    if is_print is True:
        print("ReliefF选择的重要特征[score>={}]:\n".format(score_ReliefF))
        print(score)
        print("特征个数：", score.size)

    # 画图
    score.plot(kind='barh', use_index=True)
    plt.xlabel("feature_importance")
    plt.ylabel("feature")

    if is_show is True:
        plt.show()

    # 返回ReliefF选择后的重要特征
    select_X_df = X_df.ix[:, score.index.tolist()]
    # print(select_X_df)
    # print(select_X_df.shape)
    return select_X_df


# KPCA降维算法
def KPCA(X_df, y, n_components=None, is_print=True, is_show=False):
    # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel = 'linear'
    kpca = KernelPCA(n_components=n_components, kernel=kernel)
    kpca.fit(X_df)

    lambdas = pd.Series(kpca.lambdas_)
    score = pd.Series(list([x / np.sum(kpca.lambdas_) for x in kpca.lambdas_]))
    cumsum_score = pd.Series(np.cumsum(score))
    score_df = pd.concat([lambdas, score, cumsum_score], axis=1)
    score_df.columns = ['lambdas', 'score', 'cum_score']
    # score = (score_df[score_df['cum_score'] <= 0.70]['score']).sort_values(ascending=True)
    # score = (score_df[score_df['score'] >= 0.01]['score']).sort_values(ascending=True)
    if is_print is True:
        print("KPCA降维后的主成分:\n", score_df)
        print("主成分个数：", len(score))

    # 画图
    score.plot(kind='barh', use_index=True)
    plt.xlabel("feature_importance")
    plt.ylabel("feature")

    if is_show is True:
        plt.show()

    # 返回降维后的X_df
    return kpca.transform(X_df)


# KPCA画图
def plot_KPCA(X_df, y):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    fig = plt.figure()
    colors = (
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0),
        (0, 0.6, 0.4), (0.5, 0.3, 0.2),)
    for i, kernel in enumerate(kernels):
        kpca = KernelPCA(n_components=None, kernel=kernel, random_state=1000)
        kpca.fit(X_df)
        X_r = kpca.transform(X_df)
        ax = fig.add_subplot(2, 2, i + 1)
        for label, color in zip(np.unique(y), colors):
            position = y == label
            ax.scatter(X_r[position, 0], X_r[position, 1], label="target=%d" % label, color=color)
            ax.set_xlabel("X[0]")
            ax.set_ylabel("X[1]")
            ax.legend(loc="best")
            ax.set_title("kernel=%s" % kernel)
    plt.suptitle("KPCA")
    plt.show()


def fea_select_main(X_df, y, n_components=None, score_ReliefF=0.01, is_print_ReliefF=True, is_print_KPCA=True,
                    is_show_KPCA=False, is_show_ReliefF=False):
    select_X_df = ReliefF(X_df, y, iter_ratio=0.7, k=5, norm='2', score_ReliefF=score_ReliefF,
                          is_print=is_print_ReliefF, is_show=is_show_ReliefF)
    decom_X_df = KPCA(select_X_df, y, n_components=n_components, is_print=is_print_KPCA, is_show=is_show_KPCA)
    # plot_KPCA(X_df, y)

    return decom_X_df, y


if __name__ == '__main__':
    X_df, y = fea_abstract_main()
    fea_select_main(X_df, y)
