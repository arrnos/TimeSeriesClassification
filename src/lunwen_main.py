import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from random import randrange
from sklearn.model_selection import train_test_split
pd.set_option('display.width', 1000)
import re
import pywt

from sklearn.decomposition import KernelPCA, PCA, IncrementalPCA, FactorAnalysis, RandomizedPCA, FastICA

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)

from src.lunwen_model import *
from src.lunwen_fea_abstract import *
from src.lunwen_model import *
from src.lunwen_fea_select_decomp import *


def test1_ReliefF():
    # 显示score>=0.01/score>=0.009时，筛选后的特征，画图，并打印结果
    X_df, y = fea_abstract_main()
    fea_select_main(X_df, y, n_components=15, score_ReliefF=0.013, is_print_ReliefF=True, is_print_KPCA=False,
                    is_show_KPCA=False,
                    is_show_ReliefF=True)


def test2_KPCA():
    n_components_list = np.arange(5, 45, 3)
    n_components_list = [30]
    score_ReliefF = 0.009
    for n_components in n_components_list:
        print("n_components={}的情况".format(n_components))
        X_df, y = fea_abstract_main()
        select_X_df, y = fea_select_main(X_df, y, n_components=n_components, score_ReliefF=score_ReliefF,
                                         is_print_ReliefF=False,
                                         is_print_KPCA=True, is_show_KPCA=True,
                                         is_show_ReliefF=False)


def test3_KPCA_GBDT_time_score():
    n_components_list = range(5, 41, 2)
    result_array = []
    score_ReliefF = 0.009
    for n_components in n_components_list:
        print("n_components={}的情况>>>".format(n_components))
        score_i = [n_components]
        X_df, y = fea_abstract_main()
        select_X_df, y = fea_select_main(X_df, y, n_components=n_components, score_ReliefF=score_ReliefF,
                                         is_print_ReliefF=False,
                                         is_print_KPCA=False, is_show_KPCA=False,
                                         is_show_ReliefF=False)

        # 划分数据集
        y = np.ravel(y)
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.15, random_state=1000)
        select_X_train, select_X_test, select_y_train, select_y_test = train_test_split(select_X_df, y, test_size=0.15,
                                                                                        random_state=1000)

        # 模型字典
        model_dict = {
            "SVM": SVM,
            "LR": LR,
            "KNN": KNN,
            "GBDT": GBDT,
        }

        grid = False
        silent = True
        init_train = True

        time_test = True
        score_test = True

        model_test = "GBDT"
        time_report = False
        score_report = False
        if n_components_list.index(n_components)==0:

            score_i.append("特征工程前")
            model_result = model_dict.get(model_test)(X_train, X_test, y_train, y_test, grid_search=grid,
                                                      init_train=init_train,
                                                      silent=silent,
                                                      time_report=time_report, score_report=score_report)
            score_i += model_result
            result_array.append(score_i)

        score_i = [n_components, "特征工程后"]
        model_result = model_dict.get(model_test)(select_X_train, select_X_test, select_y_train, select_y_test,
                                                  grid_search=grid,
                                                  init_train=init_train,
                                                  silent=silent,
                                                  time_report=time_report, score_report=score_report)
        score_i += model_result
        result_array.append(score_i)
    result_pd = pd.DataFrame(result_array)
    result_pd.columns = ['n_components', '对比状态', '模型训练时间(ms)', 'precision', 'recall', 'f1', 'accuracy', 'roc_auc']
    print(result_pd)
    return result_pd


    # # 时间性能测试
    # if time_test is True:
    #     model_test = ["GBDT"]
    #     time_report = True
    #     score_report = False
    #     print("#####################  时间性能测试   #####################")
    #     print("\n未进行特征选择前：")
    #     score_i.append("特征工程前")
    #     for model in model_test:
    #         model_result = model_dict.get(model)(X_train, X_test, y_train, y_test, grid_search=grid,
    #                                              init_train=init_train,
    #                                              silent=silent,
    #                                              time_report=time_report, score_report=score_report)
    #         score_i += model_result
    #         result_array.append(score_i)
    #     print("\n进行特征选择后：")
    #     score_i = [n_components, "特征工程后"]
    #     for model in model_test:
    #         model_result = model_dict.get(model)(select_X_train, select_X_test, select_y_train, select_y_test, grid_search=grid,
    #                               init_train=init_train,
    #                               silent=silent,
    #                               time_report=time_report, score_report=score_report)
    #         score_i += model_result
    #         result_array.append(score_i)
    # if score_test is True:
    #     model_test = ["LR", "SVM", "GBDT"]
    #
    #     time_report = False
    #     score_report = True
    #     # 训练并测试
    #     print("#####################   分类器对比测试   #####################")
    #     for model in model_test:
    #         print("\n进行特征选择前的性能：")
    #         model_dict.get(model)(X_train, X_test, y_train, y_test, grid_search=grid, init_train=init_train,
    #                               silent=silent,
    #                               time_report=time_report, score_report=score_report)
    #
    #         print("\n进行特征选择后的性能：")
    #         model_dict.get(model)(select_X_train, select_X_test, select_y_train, select_y_test, grid_search=grid,
    #                               init_train=init_train,
    #                               silent=silent,
    #                               time_report=time_report, score_report=score_report)


def test4_fenlei_model():
    X_df, y = fea_abstract_main()
    n_components=17
    score_ReliefF=0.009
    select_X_df, y = fea_select_main(X_df, y, n_components=n_components, score_ReliefF=score_ReliefF,
                                     is_print_ReliefF=False,
                                     is_print_KPCA=False, is_show_KPCA=False,
                                     is_show_ReliefF=False)    # 划分数据集
    y = np.ravel(y)
    # X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.15, random_state=1000)
    select_X_train, select_X_test, select_y_train, select_y_test = train_test_split(select_X_df, y, test_size=0.15,
                                                                                    random_state=1000)
    # 模型字典
    model_dict = {
        "SVM": SVM,
        "LR": LR,
        "RF":RF,
        "Linear_SVM":Linear_SVM,
        "RBF_SVM":RBF_SVM,
        "Poly_SVM":Poly_SVM,
        "Sigmoid_SVM":Sigmoid_SVM,
        "GBDT": GBDT,
    }

    grid = False
    silent = True
    init_train = True

    time_test = True
    score_test = True

    model_test = ["Linear_SVM","RBF_SVM","Poly_SVM","Sigmoid_SVM",'RF',"GBDT"]
    time_report = False
    score_report = False

    result_array=[]
    for model in model_test:
        score_i = [model]
        model_result = model_dict.get(model)(select_X_train, select_X_test, select_y_train, select_y_test,
                                              grid_search=grid,
                                              init_train=init_train,
                                              silent=silent,
                                              time_report=time_report, score_report=score_report)
        print(model_result)
        score_i += model_result
        result_array.append(score_i)


    result_pd = pd.DataFrame(result_array)
    result_pd.columns = ['模型', '模型训练时间(ms)', 'precision', 'recall', 'f1', 'accuracy', 'roc_auc']
    print(result_pd)
    return result_pd




if __name__ == '__main__':
    # test1_ReliefF()
    # test2_KPCA()
    # test3_KPCA_GBDT_time_score()
    test4_fenlei_model()
