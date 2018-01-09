from src import bcu_data_pred
from src.bcu_model import *
from sklearn.model_selection import train_test_split
import numpy as np


def bcu_model_train(model="LR", grid=False, silent=True, init_train=True, save=True):
    # 读取原始数据
    X_df_list, y = bcu_data_pred.data_read()

    # 时间序列统计特征
    X = bcu_data_pred.feature_extraction(X_df_list)

    y = np.ravel(y)

    # 将label_data顺序打散
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.65, random_state=1000)
    # 模型字典
    model_dict = {
        "SVM": SVM,
        "LR": LR,
        "KNN": KNN,
        "GBDT": GBDT,
        "XGB": XGBoost
    }

    # 训练并测试
    model_dict.get(model)(X_train, X_test, y_train, y_test, grid_search=grid, model_save=save,
                          init_train=init_train, silent=silent)


if __name__ == '__main__':
    for model in ['LR',"GBDT"]:
        bcu_model_train(model=model, silent=True, init_train=True, save=True, grid=False, )
