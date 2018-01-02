import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from numpy import linalg as la
import re
from PyEMD import EMD, EMD2d, EEMD
from pywt import *

os.chdir("/Users/arrnos/PycharmProjects/TimeSeriesClassification/")
feature_path = "MovementAAL/dataset"


# 读取数据
def data_read():
    # load label data
    y_df = pd.read_csv("MovementAAL/dataset/MovementAAL_target.csv")
    y = y_df.iloc[:, -1].replace({-1: 0})

    # load feature data
    X_file_list = filter(lambda x: x.startswith("MovementAAL_RSS_"), os.listdir(feature_path))
    X_file_list = sorted(X_file_list, key=lambda x: int(re.findall('\d+', x)[0]))
    X_df_list = []
    for file in X_file_list:
        X_df = pd.read_csv(feature_path + '/' + file, delimiter=',')
        X_df_list.append(X_df)
    return X_df_list, y


# SVD feature 构造
def SVD_feature(X_df_list, y):
    SVD_feature_list = []
    for X_matrix in X_df_list:
        U, sigma, VT = la.svd(X_matrix)
        feature = VT.reshape(1, -1)[0][:4]
        SVD_feature_list.append(feature)
    # np.savetxt('MovementAAL/dataset/feature/EMD_SVD_feature.csv',SVD_feature_list,fmt='%s',delimiter=',')
    X_SVD_feature = pd.DataFrame(SVD_feature_list)
    # return pd.concat([feature_df, target.ix[:, [-1]]], axis=1)
    return X_SVD_feature, y


# EMD feature 构造
def EMD_feature(X_df_list, y):
    columns = X_df_list[0].columns
    EMD_df_list = []
    for i, X_matrix in enumerate(X_df_list):
        # 一、EMD 特征提取 训练过程
        # t = np.arange(len(X_matrix))
        # new_EMD_df = pd.DataFrame(index=range(len(X_matrix)),columns=columns)
        # for col_i in columns:
        #     s = X_matrix[col_i].values
        #     IMF = EEMD().eemd(s, t)
        #     if len(IMF)>1:
        #         imf2 = pd.DataFrame(IMF[1])
        #     else:
        #         imf2 = pd.DataFrame(IMF[0])
        #     new_EMD_df[col_i]=imf2

        # print(type(imf2),type(IMF),IMF.shape)
        # 画图
        # plt.subplot(2, 1, 1)
        # plt.plot(s, 'r')
        # plt.title("Input signal: data0_%s" % col_i)
        # plt.subplot(N, 1, 2)
        # plt.plot(imf2, 'g')
        # plt.title("Input signal: imf_2")
        # plt.tight_layo uuuut()
        # plt.show()
        # new_EMD_df.to_csv('MovementAAL/dataset_EMD/MovementAAL_EMD_{}.csv'.format(i),index=None)
        # EMD_df_list.append(new_EMD_df)
        # print("正在处理第{}个文件".format(i))

        # 二、从文件读取
        new_EMD_df = pd.read_csv('MovementAAL/dataset_EMD/MovementAAL_EMD_{}.csv'.format(i))
        EMD_df_list.append(new_EMD_df)
    return EMD_df_list, y


# EMD_SVD feature 构造
def EMD_SVD_feature(X_list, y):
    EMD_df_list, y = EMD_feature(X_list, y)
    X_SVD_feature, y = SVD_feature(EMD_df_list, y)
    return X_SVD_feature, y


# 小波变换 feature构造
def WAV_feature(X_df_list, y):
    columns = X_df_list[0].columns
    WAV_feature_list = []
    for X_matrix in X_df_list:
        new_EMD_df = pd.DataFrame(columns=columns)
        for col_i in columns:
            X_col_i = X_matrix[col_i].values
            cA, cD = dwt(X_col_i, 'db2')
            print(cA.shape, cD.shape)
            cA2, cD2, cD1 = wavedec(X_col_i, 'db1', level=2)
            print(cA2.shape, cD2.shape, cD1.shape)


# 时间序列统计 feature构造
def ts_stats_feature(X_df_list, y):
    feature_list =[]
    columns = []
    for X_matrix in X_df_list:
        fea_line, columns = stats_feature(X_matrix)
        feature_list.append(fea_line)
    feature_df = pd.DataFrame(feature_list,columns=columns)
    # feature_df.to_csv('MovementAAL/dataset/feature/ts_corr_cov_stat_feature.csv',index=None)
    return feature_df, y


def stats_feature(x):
    # 时间序列特征
    ts_fea_list = [x.count(), x.mean(), x.min(),
                   x.quantile(.25), x.median(),
                   x.quantile(.75), x.max(),
                   x.mad(), x.var(), x.std(),
                   x.skew(), x.kurt()]

    ts_df_columns = ['Count', 'Mean', 'Min',
                     'Q1', 'Median', 'Q3',
                     'Max', 'Mad', 'Var',
                     'Std', 'Skew', 'Kurt']

    ts_df = pd.DataFrame(ts_fea_list, index=ts_df_columns)
    ts_fea = list(ts_df.values.T.reshape(1, -1)[0])

    # 时间序列特征faltten
    x_df_columns = ['RSS_1', 'RSS_2', 'RSS_3', 'RSS_4']
    ts_index = []
    for df_col in x_df_columns:
        ts_index.append([x + '_' + df_col for x in ts_df_columns])
    ts_columns = list(np.array(ts_index).reshape(1, -1)[0])

    # 变量相关关系特征
    # corr相关系数
    corr = x.corr()
    corr_triu = np.triu(corr.values, 1)
    corr_fea = [x for x in corr_triu.reshape(1, -1)[0] if x != 0]
    corr_columns = ["corr_" + str(x) for x in range(len(corr_fea))]
    # cov协方差
    cov = x.cov()
    cov_triu = np.triu(cov.values)
    cov_fea = [x for x in cov_triu.reshape(1, -1)[0] if x != 0]
    cov_columns = ["cov_" + str(x) for x in range(len(cov_fea))]
    fea_list = ts_fea+corr_fea+cov_fea
    columns = ts_columns+corr_columns+cov_columns
    return fea_list,columns

if __name__ == '__main__':
    X_df_list, y = data_read()
    # X_SVD_feature, y = EMD_SVD_feature(X_df_list,y)
    # print(X_SVD_feature)
    # WAV_feature(X_df_list,y)
    ts_stats_feature(X_df_list, y)
