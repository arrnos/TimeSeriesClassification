from src import bcu_data_pred
from src.bcu_model import *
from sklearn.model_selection import train_test_split
from src import bcu_mysql
from sklearn.externals import joblib
import numpy as np
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows',None)


def bcu_model_detect(model="LR",):

    # 加载数据
    # 从数据库查询某车次的数据，返回DataFrame,并通过滑动窗口继续分裂
    test_tid = 1
    test_line_df = bcu_mysql.mysql_query(test_tid)
    line_part_list = bcu_mysql.df_slide_window_part(test_line_df, window_len=80, step=10)
    X_test = line_part_list
    # 特征工程
    X_test= bcu_data_pred.feature_extraction(X_test)

    model= joblib.load("MovementAAL/bcu_data/model/{}_model.model".format(model))
    y_pred = model.predict(X_test)
    y_predprob = model.predict_proba(X_test)[:, 1]
    y_rs = np.where(y_predprob>0.75,y_pred,-1)
    print(y_rs)

if __name__ == '__main__':
    # for model in ['LR',"GBDT"]:
    for model in ['LR']:
        bcu_model_detect(model=model)