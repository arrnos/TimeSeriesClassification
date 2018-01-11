from src import bcu_data_pred
from src.bcu_model import *
from sklearn.model_selection import train_test_split
from src import bcu_mysql
from sklearn.externals import joblib
import numpy as np
import  datetime
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows',None)


def bcu_model_detect(model="LR", t_id = 3,window_len=200, step=100,prob=0.9):

    # 加载数据
    # 从数据库查询某车次的数据，返回DataFrame,并通过滑动窗口继续分裂
    test_line_df = bcu_mysql.query_lineData_by_tID(t_id)

    # test_line_df  = pd.read_excel("/Users/arrnos/PycharmProjects/TimeSeriesClassification/MovementAAL/bcu_data/bcu_mysql_table/J3_detail.xlsx")
    # test_line_df = test_line_df.iloc[:,1:]


    line_part_list,part_time_list = bcu_mysql.df_slide_window_part(test_line_df, window_len=window_len, step=step)
    X_test = line_part_list
    # 特征工程
    X_test= bcu_data_pred.feature_extraction(X_test)

    model= joblib.load("MovementAAL/bcu_data/model/{}_model.model".format(model))
    y_pred = model.predict(X_test)
    y_predprob0 = model.predict_proba(X_test)[:, 0]
    y_predprob1 = model.predict_proba(X_test)[:, 1]
    # print(list(zip(y_pred,y_predprob0,y_predprob1)))
    conditon = [x or y for x,y in zip(y_predprob0 > prob , y_predprob1>prob)]
    y_rs = list(np.where(conditon,y_pred,-1)) # 设置预测概率，过滤不合理的分类

    # （time,ylabel）结合
    time_ylabel_df = pd.DataFrame(np.array([part_time_list,y_rs]).T,columns=['bcu_fault_time','ylabel'])
    # 过滤正常类
    filter_index = time_ylabel_df['ylabel']!=-1
    time_ylabel_filter = time_ylabel_df[filter_index] # 异常时间点和故障类别
    kilometer = time_ylabel_filter.iloc[:,0].apply(lambda x :(bcu_mysql.query_kilo_by_time(x,t_id))[1])
    kilometer = kilometer[kilometer>0] #过滤负数和空的公里标
    if len(kilometer)==0:
        return time_ylabel_filter

    jwd=kilometer.apply(lambda x:bcu_mysql.query_jwd_by_kilometer(x))
    l1=jwd.map(lambda x:x[0])
    l2 = jwd.map(lambda x:x[1])
    time_ylabel_filter.insert(2,'kilometer',kilometer)
    time_ylabel_filter.insert(3,'l1',l1)
    time_ylabel_filter.insert(4,'l2',l2)


    # # 查看对应时间的bcu数据
    # part_bcu_list = []
    # time_ylabel_filter['bcu_fault_time'].apply(lambda time:part_bcu_list.append(line_part_list[part_time_list.index(time)]))
    # print(part_bcu_list)

    return time_ylabel_filter

if __name__ == '__main__':
    # for model in ['LR',"GBDT"]:
    for t_id in [0]:
        for model in ['GBDT']:
            rs = bcu_model_detect(model=model,t_id=t_id,window_len=100, step=70,prob=0.99)
            print("t_id:{},model:{}".format(t_id,model))
            print(rs)
