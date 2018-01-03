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


def LR(X_train, X_test, y_train, y_test, grid_search=False, init_train=True, silent=True):
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
        if init_train is False:
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


def SVM(X_train, X_test, y_train, y_test, grid_search=False, init_train=True, silent=False):
    if grid_search is False:
        model = SVC(probability=True)
        if init_train is False:
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


def KNN(X_train, X_test, y_train, y_test, grid_search=False, init_train=True, silent=False):
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


def GBDT(X_train, X_test, y_train, y_test, grid_search=False, init_train=True, silent=False):
    """
    调参请看：http://www.cnblogs.com/pinard/p/6143927.html
    """
    if grid_search is False:

        model = GradientBoostingClassifier(random_state=100)

        if init_train is False:
            model.set_params(learning_rate=0.1, n_estimators=100, random_state=100, min_samples_leaf=16,
                             min_samples_split=2)

        model.fit(X_train, y_train)
        print("\n\n*********************  GBDT 预测：  ********************* \n")
        if silent is False:
            print("model : \n", model)
            print("n_estimators : ", model.n_estimators)
            # print("n_features_ : ",model.n_features_)
            # print("feature_importances_ : ",model.feature_importances_ )
        y_pred = model.predict(X_test)
        y_predprob = model.predict_proba(X_test)[:, 1]

        print('\n', classification_report(y_test, y_pred))
        print("Accuracy : %.4g" % accuracy_score(y_test, y_pred))
        print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob))
    else:
        print("\n\n********************* GBDT Grid Search：  ********************* \n")
        # Set the parameters by cross-validation
        """
        *** 树参数 ***
        
        1. max_ depth 
        树的最大深度，可以控制过度拟合，因为分类树越深就越可能过度拟合。
        2. min_ samples_split：
        树中一个节点所需要用来分裂的最少样本数。可以避免过度拟合。
        3. min_ samples_leaf 
        定义了树中终点节点所需要的最少的样本数，也可以用来防止过度拟合。在不均等分类问题中(imbalanced class problems)，一般这个参数需要被设定为较小的值，因为大部分少数类别（minority class）含有的样本都比较小。
        4. max_ features 
        决定了用于分类的特征数，根据经验一般选择总特征数的平方根就可以，也可以CV尝试总特征数的30%-40%.
        
        *** boost参数 ***
        
        1. learning_ rate 
        决定着每一个树对于最终结果的影响，控制着每次更新的幅度，较小的learning rate使得模型对不同的树更加稳健。
        2. n_ estimators 
        需要使用到的决定树的数量，在有较多决定树时能保持稳健，但还是可能发生过度拟合。，需要针对learning rate用CV值检验。
        3. subsample
        训练每个决定树所用到的子样本占总样本的比例，而对于子样本的选择是随机的，用稍小于1的值能够使模型更稳健，因为这样减少了方差。
        """

        step = 6
        if step == 1:
            param_test = {'n_estimators': range(95, 115, 1)}
            model1 = GradientBoostingClassifier(random_state=100)
            model_select(X_train, X_test, y_train, y_test, model1, param_test)
        if step == 2:
            param_test = {'max_depth': range(2, 5, 1), 'min_samples_split': range(16, 25, 1)}
            model1 = GradientBoostingClassifier(random_state=100, n_estimators=100)
            model_select(X_train, X_test, y_train, y_test, model1, param_test)
        if step == 3:
            param_test = {'min_samples_split': range(2, 10, 2), 'min_samples_leaf': range(10, 20, 2)}
            model1 = GradientBoostingClassifier(random_state=100, n_estimators=100, max_depth=3)
            model_select(X_train, X_test, y_train, y_test, model1, param_test)
        if step == 4:
            param_test = {'max_features': range(10, 60, 1)}
            model1 = GradientBoostingClassifier(random_state=100, n_estimators=97, max_depth=3, min_samples_leaf=16,
                                                min_samples_split=2)
            model_select(X_train, X_test, y_train, y_test, model1, param_test)
        if step == 5:
            param_test = {'subsample': np.arange(0.5, 1.0, 0.05)}
            model1 = GradientBoostingClassifier(random_state=100, n_estimators=100, max_depth=3, min_samples_leaf=16,
                                                min_samples_split=2, max_features=38)
            model_select(X_train, X_test, y_train, y_test, model1, param_test)
        if step == 6:
            param_test = [{'learning_rate': [0.05], 'n_estimators': [200]},
                          {'learning_rate': [0.01], 'n_estimators': [1000]},
                          {'learning_rate': [0.005], 'n_estimators': [2000]}]
            model1 = GradientBoostingClassifier(random_state=100, max_depth=3, min_samples_leaf=16, min_samples_split=2,
                                                max_features=38, subsample=0.85)
            model_select(X_train, X_test, y_train, y_test, model1, param_test)



            # XGBoost 分类器模型


def XGBoost(X_train, X_test, y_train, y_test, grid_search=False, init_train=True, silent=False):
    """
    调参参考：http://blog.csdn.net/han_xiaoyang/article/details/52665396

    默认初始值：
    1、learning_rate[默认0.3]   典型值为0.01-0.2   初始值：0.1
    2、min_child_weight[默认1]  决定最小叶子节点样本权重和 分类不平衡问题，要减小
    3、max_depth[默认6]   典型值：3-10 初始值：5
    4、gamma[默认0] Gamma指定了节点分裂所需的最小损失函数下降值。典型值: [0.1,0.2]
    5、subsample[默认1]  典型值：0.5-1   推荐值：0.8
    6、colsample_bytree[默认1]  典型值：0.5-1   推荐值：0.8
    7、lambda[默认1]   权重的L2正则化项。
    8、alpha[默认1]    权重的L1正则化项。
    9、scale_pos_weight[默认1]  在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
    """
    if grid_search is False:

        # 默认参数设置
        params_default = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',  # 多分类的问题
            'num_class': 2,  # 类别数，与 multisoftmax 并用
            # 'max_depth': 5,  # 构建树的深度，越大越容易过拟合
            # 'min_child_weight': 1,
            # 'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            # 'subsample': 0.8,  # 随机采样训练样本
            # 'colsample_bytree': 0.8,  # 生成树时进行的列采样
            # 'lambda': 1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            # 'alpha': 0.1,  # 控制模型复杂度的权重值的L1正则化项参数，参数越大，模型越不容易过拟合。
            'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
        }

        # 定义数据
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=500)
        xgb_valid = xgb.DMatrix(X_valid, label=y_valid)
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_test = xgb.DMatrix(X_test, y_test)

        watchlist = [(xgb_train, 'train'), (xgb_valid, 'val')]
        num_rounds = 5000  # 迭代次数
        early_stopping = 100

        # 优化参数
        params = {}

        if init_train is True:
            params = params_default

        # 训练模型并保存
        model = xgb.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=early_stopping)
        # 获取验证集合结果
        model.save_model('MovementAAL/model/xgb.model')  # 用于存储训练出的模型
        print("best best_ntree_limit:", model.best_ntree_limit)
        y_pred = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)

        print('\n', classification_report(y_test, y_pred))
        print("Accuracy : %.4g" % accuracy_score(y_test, y_pred))

    else:
        print("\n\n********************* XGBoost Grid Search：  ********************* \n")
        # Set the parameters by cross-validation
        # 初始化模型
        model = XGBClassifier(learning_rate=0.1,
                              max_depth=5, min_child_weight=1, gamma=0,
                              colsample_bytree=0.8, subsample=0.8,
                              objective='binary:logistic', random_state=100)
        # 调参步骤
        step = 7

        if step == 1:
            # n_estimators 调优

            param_test = {'n_estimators': range(50, 80, 1)}

            model_select(X_train, X_test, y_train, y_test, model, param_test)

        if step == 2:
            # max_depth 和 min_weight 调优

            param_test = {'max_depth': range(3, 6, 1), 'min_child_weight': np.arange(0.1, 0.3, 0.02)}

            model.set_params(n_estimators=60)

            model_select(X_train, X_test, y_train, y_test, model, param_test)

        if step == 3:
            # gamma参数调优

            param_test = {'gamma': [0, 0.1, 0.01]}

            model.set_params(n_estimators=60, max_depth=4, min_child_weight=0.2)

            model_select(X_train, X_test, y_train, y_test, model, param_test)

        if step == 4:
            # subsample 和 colsample_bytree 调优

            param_test = {'subsample': np.arange(0.5, 0.96, 0.05), 'colsample_bytree': np.arange(0.5, 0.96, 0.05)}

            model.set_params(n_estimators=60, max_depth=4, min_child_weight=0.2, gamma=0)

            model_select(X_train, X_test, y_train, y_test, model, param_test)

        if step == 5:
            # reg_alpha调优

            # param_test = {'reg_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]}

            param_test = {'reg_alpha': np.arange(0.005, 0.15, 0.001)}

            model.set_params(n_estimators=60, max_depth=4, min_child_weight=0.2, gamma=0, subsample=0.6,
                             colsample_bytree=0.8)

            model_select(X_train, X_test, y_train, y_test, model, param_test)

        if step == 6:
            # reg_lambda调优

            # param_test = {'reg_lambda': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]}

            param_test = {'reg_lambda': np.arange(0.1, 2, 0.1)}

            model.set_params(n_estimators=60, max_depth=4, min_child_weight=0.2, gamma=0, subsample=0.6,
                             colsample_bytree=0.8, reg_alpha=0.01)

            model_select(X_train, X_test, y_train, y_test, model, param_test)

        if step == 7:
            # 学习率调优learning_rate
            n = 60
            param_test = [{'learning_rate': [0.1], 'n_estimators': [n]},
                          {'learning_rate': [0.05], 'n_estimators': [n * 2]},
                          {'learning_rate': [0.01], 'n_estimators': [n * 10]},
                          {'learning_rate': [0.005], 'n_estimators': [n * 20]}]

            model.set_params(max_depth=4, min_child_weight=0.2, gamma=0, subsample=0.6,
                             colsample_bytree=0.8, reg_alpha=0.01, reg_lambda=1)

            model_select(X_train, X_test, y_train, y_test, model, param_test)


# model_select
def model_select(X_train, X_test, y_train, y_test, model, grid_param, cv=5):
    # scores = ['precision', 'recall','accuracy','f1','roc_auc']
    scores = ['roc_auc']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, grid_param, cv=cv, n_jobs=4, iid=False,
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
        # print("opt model is :")
        # print(clf)
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        print()
        y_predprob = clf.predict_proba(X_test)[:, 1]
        acc_score = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_predprob)
        print("Accuracy : %.4g" % acc_score)
        print("AUC Score (Train): %f" % auc_score)


def rocCurve(y_test, y_pred, y_predprob):
    # 绘制roc曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_predprob, pos_label=1.0)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', linewidth=lw,
             label='ROC curve (area = %0.4f)' % roc_auc_score(y_test, y_pred))
    plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
