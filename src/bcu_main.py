from src import bcu_data_pred
from src.model import *
from src.feature_select import *
import numpy as np
from sklearn.model_selection import train_test_split

# 一、读取原始数据
X_df_list, y = bcu_data_pred.data_read()

# 二、特征提取

# SVD 数据
# X, y = SVD_feature(X_df_list, y)

# EMD_SVD 数据
# X, y = EMD_SVD_feature(X_df_list,y)

# 时间序列统计特征
X, y = bcu_data_pred.feature_extraction(X_df_list, y)

# X = pd.read_csv("MovementAAL/dataset/feature/ts_stat_feature.csv",sep=',',header=0)


# 三、特征选择
# for k in range(35,50,1):
#     X1,y1 = univariate_feature_selection(X,y,k=k,selent=True)
#     print("特征数量：",X1.shape[1])

# k=39时，Accuracy: 0.8958
# AUC Score(Train): 0.910714


# 三、特征降维
# for n_com in range(10,20,1):
#     print("n_components : ",n_com)
#     X1,y1=decom_ica(X,y,n_components=n_com)

# 四、数据输出
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1000)

# 模型字典
model_dict = {
"SVM": SVM,
"LR": LR,
"KNN": KNN,
"GBDT": GBDT,
"XGB": XGBoost
}

# 待测试模型列表
# model_test = ["LR","SVM", "KNN","GBDT"]
model_test = ["GBDT"]
# grid=True
grid = False
silent = True
# init_train = False
init_train = True

# 训练并测试
for model in model_test:
    model_dict.get(model)(X_train, X_test, y_train, y_test, grid_search=grid, init_train=init_train, silent=silent)
