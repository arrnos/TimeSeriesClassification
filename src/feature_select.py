from src.data_pred import *

from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LassoCV


def univariate_feature_selection(X, y, k=None, percentile=None, selent=False):
    """
    单变量特征选择（统计测试方法）
    原理：根据每个特征与label特征的相关性进行rank，选择出一定数量或比较的最优特征
    主要方法：
    SelectKBest选择排名排在前n个的变量
    SelectPercentile 选择排名排在前n%的变量

    回归问题: f_regression, mutual_info_regression
    分类问题: chi2, f_classif, mutual_info_classif
    """

    if k is not None and percentile is None:
        selector = SelectKBest(k=k)
        new_X = selector.fit_transform(X, y)
        if selent is False:
            print("fea_num:", k, ", fea_num:", new_X.shape[1])
            print("fea_col:", new_X.columns)
        return new_X, y
    elif k is None and percentile is not None:
        selector = SelectPercentile(percentile=percentile)
        new_X = selector.fit_transform(X, y)
        if selent is False:
            print("fea_perc:", k, ", fea_num:", new_X.shape[1])
            print("fea_col:", new_X.columns)
        return new_X, y
    elif k is None and percentile is None:
        print("params error! elect one from k and percentile")


def rfe(X, y, n_features_to_select, step=1, selent=True):
    """
    递归特征消除
    依赖：a estimator which has a coef_ attribute or feature_importances_ attribute
    适用模型：SVM,LassoCV等
    原理：select features by recursively considering smaller and smaller sets of features.
    步骤：First, the estimator is trained on the initial set of features and the importance of
    each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
    """
    svc = SVC(kernel='linear')
    rfe = RFE(estimator=svc, n_features_to_select=n_features_to_select, step=step)
    rfe.fit(X, y)
    print("rfe.n_features_:{}".format(rfe.n_features_))
    print("rfe.support_:{}".format(rfe.support_))
    print("rfe.ranking:{}".format(rfe.ranking_))
    X_new = rfe.transform(X)
    return X_new, y


def select_from_model(X, y, method="tree", selent=True):
    """
    适用范围： any estimator that has a coef_ or feature_importances_ attribute after fitting
    原理：The features are considered unimportant and removed, if the corresponding coef_ or feature_importances_ values
    are below the provided threshold parameter.
    两种模型：
    ***  L1-based feature selection  ***
    In particular, sparse estimators useful for this purpose are the linear_model.Lasso for regression, and of
    linear_model.LogisticRegression and svm.LinearSVC for classification.

    ***  tree model ***
    see the sklearn.tree module and forest of trees in the sklearn.ensemble module) can be used to compute feature importances,
    which in turn can be used to discard irrelevant features
    """
    if method == "L1":
        model = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
        selector = SelectFromModel(model, prefit=True)
        X_new = selector.transform(X)
        return X_new, y
    if method == "tree":
        clf = ExtraTreesClassifier()
        clf = clf.fit(X, y)
        print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)
        print(X_new.shape)
        return X_new,y

if __name__ == '__main__':
    X_df_list, y = data_read()
    X, y = ts_stats_feature(X_df_list, y)
    # select_percentile(X, y, percentile=90)
    select_from_model(X, y)
