import numpy as np
import pandas as pd
import os

from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


def LR(X_train, X_test, y_train, y_test, param_grid=None, model_print=None):
    print("\n\n*********************   LR 预测：  ********************* \n")

    model = LogisticRegressionCV()

    if param_grid is None:
        model.fit(X_train, y_train)
    else:
        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid.fit(X_train, y_train)
        print(grid)
        # summarize the results of the grid search
        print(grid.best_score_)
        print(grid.best_estimator_)

    if model_print is True:
        print("model : \n", model)
    y_pred = model.predict(X_test)
    y_predprob = model.predict_proba(X_test)[:, 1]
    print('\n', metrics.classification_report(y_test, y_pred))
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))


def SVM(X_train, X_test, y_train, y_test, param_grid=None, model_print=None):
    # fit a SVM model to the data
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    print("\n\n********************* SVM  预测：  ********************* \n")
    if model_print is True:
        print("model : \n", model)
    y_pred = model.predict(X_test)
    y_predprob = model.predict_proba(X_test)[:, 1]
    print('\n', metrics.classification_report(y_test, y_pred))
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))


def CART(X_train, X_test, y_train, y_test, param_grid=None, model_print=None):
    # fit a CART model to the data
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    print("\n\n*********************  CART 预测：  ********************* \n")
    if model_print is True:
        print("model : \n", model)
    y_pred = model.predict(X_test)
    y_predprob = model.predict_proba(X_test)[:, 1]
    print('\n', metrics.classification_report(y_test, y_pred))
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))


def KNN(X_train, X_test, y_train, y_test, param_grid=None, model_print=None):
    # fit a k-nearest neighbor model to the data
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    print("\n\n*********************  KNN 预测：  ********************* \n")
    if model_print is True:
        print("model : \n", model)
    y_pred = model.predict(X_test)
    y_predprob = model.predict_proba(X_test)[:, 1]
    print('\n', metrics.classification_report(y_test, y_pred))
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))


def NB(X_train, X_test, y_train, y_test, param_grid=None, model_print=None):
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("\n\n*********************  NB 预测：  ********************* \n")
    if model_print is True:
        print("model : \n", model)
    y_pred = model.predict(X_test)
    y_predprob = model.predict_proba(X_test)[:, 1]
    print('\n', metrics.classification_report(y_test, y_pred))
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))


# GBDT 分类器模型
def GBDT(X_train, X_test, y_train, y_test, param_grid=None, model_print=None):
    model = GradientBoostingClassifier(random_state=10)
    model.fit(X_train, y_train)
    print("\n\n*********************  GBDT 预测：  ********************* \n")
    if model_print is True:
        print("model : \n", model)
    y_pred = model.predict(X_test)
    y_predprob = model.predict_proba(X_test)[:, 1]
    print('\n', metrics.classification_report(y_test, y_pred))
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))


# XGBoost 分类器模型
def XGBoost(X_train, X_test, y_train, y_test, param_grid=None, model_print=None):
    # 参数设置
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',  # 多分类的问题
        'num_class': 2,  # 类别数，与 multisoftmax 并用
        # 'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        # 'max_depth': 12,  # 构建树的深度，越大越容易过拟合
        # 'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        # 'subsample': 0.7,  # 随机采样训练样本
        # 'colsample_bytree': 0.7,  # 生成树时进行的列采样
        # 'min_child_weight': 3,
        # # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
        # 'eta': 0.01,  # 如同学习率
        # 'seed': 1000,
        # 'nthread': 7,  # cpu 线程数
    }

    X_train, X_valid,y_train,y_valid = train_test_split(X_train,y_train, test_size=0.15, random_state=500)
    xgb_valid = xgb.DMatrix(X_valid, label=y_valid)
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test,y_test)

    num_rounds = 5000  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_valid, 'val')]

    # 训练模型并保存
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(params,xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
    model.save_model('MovementAAL/model/xgb.model')  # 用于存储训练出的模型
    print("best best_ntree_limit:", model.best_ntree_limit)
    y_pred = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)

    print('\n', metrics.classification_report(y_test, y_pred))
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))


