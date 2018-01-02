import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


def LR(X_train, X_test, y_train, y_test, grid_search=False, silent=True):
    """
    (一) penalty惩罚项 : ‘l1’ or ‘l2’, 默认: ‘l2’
    注：在调参时如果我们主要的目的只是为了解决过拟合，一般penalty选择L2正则化就够了。但是如果选择L2正则化发现还是过拟合，即预测效果差的时候，就可以考虑L1正则化。另外，如果模型的特征非常多，我们希望一些不重要的特征系数归零，从而让模型系数稀疏化的话，也可以使用L1正则化。

    (二) solver优化方法
    （1）liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
    （2）lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
    （3）newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
    （4）sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候，SAG是一种线性收敛算法，这个速度远比SGD快。
    注：从上面的描述可以看出，newton-cg, lbfgs和sag这三种优化算法时都需要损失函数的一阶或者二阶连续导数，因此不能用于没有连续导数的L1正则化，只能用于L2正则化。而liblinear通吃L1正则化和L2正则化。

"""
    if grid_search is False:

        print("\n\n*********************   LR 预测：  ********************* \n")

        model = LogisticRegression()
        model.set_params(C=61, penalty='l1', solver='liblinear')
        model.fit(X_train, y_train)
        if silent is False:
            print("model : \n", model)
        y_pred = model.predict(X_test)
        y_predprob = model.predict_proba(X_test)[:, 1]
        print('\n', classification_report(y_test, y_pred))
        print("Accuracy : %.4g" % accuracy_score(y_test, y_pred))
        print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob))
    else:
        print("\n\n********************* LR Grid Search：  ********************* \n")
        # Set the parameters by cross-validation
        # {penalty=’l2’, dual = False, tol = 0.0001, C = 1.0, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None, solver =’liblinear’, max_iter = 100, multi_class =’ovr’, verbose = 0, warm_start = False}

        grid_param = [{'penalty': ['l1', 'l2'], 'C': range(1, 200, 10), 'solver': ['liblinear']}]
        model = LogisticRegression(max_iter=10000)
        model_select(X_train, X_test, y_train, y_test, model, grid_param)


def SVM(X_train, X_test, y_train, y_test, grid_search=False, silent=False):
    if grid_search is False:
        model = SVC(probability=True)
        model.set_params(kernel='rbf', C=200, gamma=0.0014)
        model.fit(X_train, y_train)
        print("\n\n********************* SVM  预测：  ********************* \n")
        if silent is False:
            print("model : \n", model)
        y_pred = model.predict(X_test)
        y_predprob = model.predict_proba(X_test)[:, 1]
        print('\n', classification_report(y_test, y_pred))
        print("Accuracy : %.4g" % accuracy_score(y_test, y_pred))
        print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob))
    else:
        print("\n\n********************* SVM Grid Search：  ********************* \n")

        # Set the parameters by cross-validation
        # grid_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
        #                 'C': [1, 10, 100, 1000]},
        #                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


        grid_param = [{'kernel': ['rbf'], 'gamma': np.arange(0.001, 0.002, 0.0002), 'C': range(50, 300, 50)}]
        model = SVC(probability=True)
        model_select(X_train, X_test, y_train, y_test, model, grid_param)


def KNN(X_train, X_test, y_train, y_test, grid_search=False, silent=False):
    if grid_search is False:
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        print("\n\n*********************  KNN 预测：  ********************* \n")
        if silent is False:
            print("model : \n", model)
        y_pred = model.predict(X_test)
        y_predprob = model.predict_proba(X_test)[:, 1]
        print('\n', classification_report(y_test, y_pred))
        print("Accuracy : %.4g" % accuracy_score(y_test, y_pred))
        print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob))
    else:
        print("\n\n********************* KNN Grid Search：  ********************* \n")
        # Set the parameters by cross-validation
        grid_params = [{'k': range(3, 15)}]
        model = KNeighborsClassifier()
        model_select(X_train, X_test, y_train, y_test, model, grid_params)


# GBDT 分类器模型
def GBDT(X_train, X_test, y_train, y_test, grid_search=False, silent=False):
    if grid_search is False:
        model = GradientBoostingClassifier(random_state=100)
        model.fit(X_train, y_train)
        print("\n\n*********************  GBDT 预测：  ********************* \n")
        if silent is False:
            print("model : \n", model)
        y_pred = model.predict(X_test)
        y_predprob = model.predict_proba(X_test)[:, 1]

        print('\n', classification_report(y_test, y_pred))
        print("Accuracy : %.4g" % accuracy_score(y_test, y_pred))
        print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob))
    else:
        print("\n\n********************* GBDT Grid Search：  ********************* \n")
        # Set the parameters by cross-validation
        grid_params = [{'k': range(3, 15)}]
        model = KNeighborsClassifier()
        model_select(X_train, X_test, y_train, y_test, model, grid_params)


# XGBoost 分类器模型
def XGBoost(X_train, X_test, y_train, y_test, silent=False):
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

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=500)
    xgb_valid = xgb.DMatrix(X_valid, label=y_valid)
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test, y_test)

    num_rounds = 5000  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_valid, 'val')]

    # 训练模型并保存
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
    model.save_model('MovementAAL/model/xgb.model')  # 用于存储训练出的模型
    print("best best_ntree_limit:", model.best_ntree_limit)
    y_pred = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)

    print('\n', classification_report(y_test, y_pred))
    print("Accuracy : %.4g" % accuracy_score(y_test, y_pred))


# model_select
def model_select(X_train, X_test, y_train, y_test, model, grid_param, cv=5):
    # scores = ['precision', 'recall','accuracy','f1','roc_auc']
    scores = ['precision', 'recall', 'accuracy']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, grid_param, cv=cv,
                           scoring='%s' % score)
        # scoring='%s_macro' % score：precision_macro、recall_macro是用于multiclass/multilabel任务的

        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        print()

        # # 绘制roc曲线
        # y_predprob = clf.predict_proba(X_test)[:, 1]
        # fpr, tpr, thresholds = roc_curve(y_test, y_predprob, pos_label=1.0)
        # plt.figure()
        # lw = 2
        # plt.plot(fpr, tpr, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % roc_auc_score(y_test,y_pred))
        # plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()
