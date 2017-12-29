from src.data_pred import *
from src import model
import numpy as np
from sklearn.cross_validation import train_test_split

# 读取原始数据
X_df_list, y = data_read()

# 特征工程

# SVD 数据
# X, y = SVD_feature(X_df_list, y)

# EMD_SVD 数据
# X, y = EMD_SVD_feature(X_df_list,y)

# 时间序列统计特征
x, y = ts_stats_feature(X_df_list, y)
X = pd.read_csv("MovementAAL/dataset/feature/ts_stat_feature.csv",sep=',',header=0)

y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1000)

# 模型字典
model_dict = {
        "SVM": model.SVM,
        "LR": model.LR,
        "CART": model.CART,
        "KNN": model.KNN,
        "NB": model.NB,
        "GBDT": model.GBDT,
        "XGB": model.XGBoost
    }

# 待测试模型列表
# model_test = ["LR","SVM", "KNN","GBDT"]
model_test = ["LR","SVM","KNN","GBDT","XGB"]

# 训练并测试
for model in model_test:
    model_dict.get(model)(X_train, X_test, y_train, y_test)
