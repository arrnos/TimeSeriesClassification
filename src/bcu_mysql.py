import pymysql
import pandas as pd
import numpy as np


def mysql_query(t_id):
    # 打开数据库连接
    db = pymysql.connect("localhost", "root", "root123", "shuohuang", charset='utf8')

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    sql = "select * from linedata where tId ={}".format(t_id)
    cursor.execute(sql)

    # 使用 fetchone() 方法获取单条数据.
    data = list(cursor.fetchall())  # 结果返回元组格式，需要转化为list,便于传入DataFrame
    columns = ['t_id', '时间', '大小闸', 'po1', '均衡风缸', '列车管',
               'po2', '预控', '闸1', '闸2', '总风缸']
    df = pd.DataFrame(data, columns=columns)
    df = df.iloc[:, 1:]

    # 关闭数据库连接
    db.close()
    return df

#滑动窗口获取长时间序列的一系列片段，用于导入分类模型进行预测
def df_slide_window_part(data_df,window_len = 80,step=10):
    line_part_list=[]
    start=0
    while(start+80<len(data_df)):
        new_part = data_df.iloc[start:start+80,:]
        start+=step
        line_part_list.append(new_part)
    return line_part_list


if __name__ == '__main__':
    df = mysql_query(1)
    df_slide_window_part(df)
