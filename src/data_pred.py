import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from numpy import linalg as la
import re
from PyEMD import EMD, EMD2d, EEMD
from pywt import *
import pywt

from sklearn.decomposition import PCA, IncrementalPCA, FactorAnalysis, RandomizedPCA, FastICA

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


# 时间序列统计 feature构造
def feature_extraction(X_df_list, type='time_freq_wave'):
    """
    :param X_df_list: X_df_list
    :param y: label
    :param type: 特征类型，Optional["time","freq","wave","time_freq","time_wave","freq_wave","time_freq_wave"]
    :return: feature_df, y
    """
    if type not in ["time","freq","wave","time_freq","time_wave","freq_wave","time_freq_wave"]:
        print('type error,type should be one of [time,freq,wave,time_freq,time_wave,freq_wave,time_freq_wave]')
        return
    #  时域特征提取
    time_feature_df = time_domain_feature(X_df_list)
    # time_feature_df.to_csv('MovementAAL/dataset/feature/time_feature.csv',index=None)

    #  频域特征提取
    freq_feature_df = freq_domain_feature(X_df_list)
    # freq_feature_df.to_csv('MovementAAL/dataset/feature/freq_feature.csv',index=None)

    #  小波能量特征提取
    wave_feature_df = wave_domain_feature(X_df_list)
    # wave_feature_df.to_csv('MovementAAL/dataset/feature/wave.csv',index=None)

    if type == "time":
        feature_df = time_feature_df
    elif type == "freq":
        feature_df = freq_feature_df
    elif type == "wave":
        feature_df = wave_feature_df
    elif type == "time_freq":
        feature_df = pd.concat([time_feature_df,freq_feature_df],axis=1)
    elif type == "time_wave":
        feature_df = pd.concat([time_feature_df,wave_feature_df],axis=1)
    elif type == "freq_wave":
        feature_df = pd.concat([freq_feature_df,wave_feature_df],axis=1)
    else:
        feature_df = pd.concat([time_feature_df,freq_feature_df,wave_feature_df],axis=1)

    # feature_df.to_csv('MovementAAL/dataset/feature/time_fre_wave_feature.csv',index=None)
    return feature_df


def time_domain_feature(X_df_list):
    # 1. 时间序列columns
    ts_df_columns = ['Count', 'Mean', 'Min',
                     'Q1', 'Median', 'Q3',
                     'Max', 'PPv', 'Mad', 'Var',  # 峰峰值 平均绝对离差 方差
                     'Std', 'Skew', 'Kurt', 'Crest_Factor']  # 均方根，峰度，偏度,波峰因子
    # 时间序列特征faltten
    x_df_columns = X_df_list[0].columns
    ts_index = []
    for df_col in x_df_columns:
        ts_index.append([x + '_' + df_col for x in ts_df_columns])
    ts_columns = list(np.array(ts_index).reshape(1, -1)[0])

    n = len(x_df_columns)
    corr_columns = ["corr_" + str(x) for x in range(n * (n - 1) // 2)]  # len(corr_fea)=len(columns)*(len(columns)-1)/2
    cov_columns = ["cov_" + str(x) for x in range(n * (n + 1) // 2)]  # len(cov_fea)

    columns = ts_columns + corr_columns + cov_columns

    # 3. 特征list
    feature_list = []

    for x in X_df_list:
        # 时间序列特征构造
        ts_fea_list = [x.count(), x.mean(), x.min(),
                       x.quantile(.25), x.median(),
                       x.quantile(.75), x.max(),
                       x.max() - x.min(), x.mad(),
                       x.var(), x.std(),
                       x.skew(), x.kurt(),
                       x.max() / x.pow(2).mean()
                       ]

        ts_df = pd.DataFrame(ts_fea_list, index=ts_df_columns)
        ts_fea = list(ts_df.values.T.reshape(1, -1)[0])

        # 变量相关关系特征
        # corr相关系数
        corr = x.corr().values
        corr_fea = []
        for i, row in enumerate(corr):
            for j, col in enumerate(row):
                if i < j:
                    corr_fea.append(col)

        # cov协方差
        cov = x.cov().values
        cov_fea = []
        for i, row in enumerate(cov):
            for j, col in enumerate(row):
                if i <= j:
                    cov_fea.append(col)

        fea_line = ts_fea + corr_fea + cov_fea
        feature_list.append(fea_line)

    ts_feature_df = pd.DataFrame(feature_list, columns=columns)
    return ts_feature_df


# 小波变换 feature构造
def wave_domain_feature(X_df_list):
    columns = X_df_list[0].columns
    wave_fea_list = []
    level = 0
    for x in X_df_list:
        x_col_fea_list = []
        for col_name in columns:
            col = x[col_name]
            col_e, level = col_wavedec(col)
            x_col_fea_list += col_e

        wave_fea_list.append(x_col_fea_list)
    coeff_columns = ['cA_%d' % level] + ['cD_%d' % i for i in range(level, 0, -1)]
    wave_fea_columns = [x + "_" + y for x in columns for y in coeff_columns]
    wave_feature_df = pd.DataFrame(wave_fea_list, columns=wave_fea_columns)
    return wave_feature_df


# 每列进行小波分解重构，得到能量值
def col_wavedec(col):
    level = 4
    wavelet = 'db1'
    col_coeffs = pywt.wavedec(col, wavelet=wavelet, level=level)
    col_e = []
    for i, coef in enumerate(col_coeffs):
        part = 'a' if i == 1 else 'd'
        col_up_xi = pywt.upcoef(part, coef, wavelet=wavelet, level=level, take=len(col))
        col_ei = np.sum([x ** 2 for x in col_up_xi])
        col_e.append(col_ei)

    return col_e, level


# 频域特征提取
def freq_domain_feature(X_df_list):
    columns = X_df_list[0].columns
    freq_fea_list = []
    for x in X_df_list:
        x_col_fea_list = []
        for col_name in columns:
            col = x[col_name]
            col_freq_fea = col_freq_feature(col)
            x_col_fea_list += col_freq_fea

        freq_fea_list.append(x_col_fea_list)
    freq_columns = ['FC', 'MSF', 'RMSF', 'VF', 'RVF', 'FVM']
    freq_fea_columns = [x + "_" + y for x in columns for y in freq_columns]
    freq_feature_df = pd.DataFrame(freq_fea_list, columns=freq_fea_columns)
    return freq_feature_df


# 每列的频域特征提取
def col_freq_feature(col):
    N = len(col)
    y = abs(np.fft.fft(col, N))
    fk = np.fft.fftfreq(N)
    y_sum = np.sum(y)
    y_mean = np.mean(y)
    y_fk_dot = np.dot(fk, y)
    y_fk2_dot = np.dot(fk ** 2, y)
    # 重心频率FC
    FC = y_fk_dot / y_sum
    # 均方频率MSF
    MSF = y_fk2_dot / y_sum
    # 均方根频率RMSF
    RMSF = np.sqrt(MSF)
    # 频率方差VF
    VF = np.dot((fk - MSF) ** 2, y)
    # 频率标准差RVF
    RVF = np.sqrt(VF)
    # 频率幅值平均值
    FVM = y_mean

    col_fea_list = [FC, MSF, RMSF, VF, RVF, FVM]

    # print("FC:{},MSF:{},RMSF:{},VF:{},RVF:{},FVM:{}".format(FC, MSF, RMSF, VF, RVF,FVM))

    return col_fea_list


# 降维
def decom_pca(feature_df, y, n_components=17):
    pca = PCA(n_components=n_components, random_state=100)
    pca.fit(feature_df, y)
    new_X = pca.transform(feature_df)
    return new_X, y


if __name__ == '__main__':
    X_df_list, y = data_read()
    # X_SVD_feature, y = EMD_SVD_feature(X_df_list,y)
    # print(X_SVD_feature)
    # WAV_feature(X_df_list,y)
    # feature_df, y = ts_stats_feature(X_df_list, y)
    # decom_pca(feature_df, y)
    # freq_domain_feature()
    x,y = feature_extraction(X_df_list, y)
    print(x)
