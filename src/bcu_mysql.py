import pymysql
import pandas as pd
import numpy as np
from datetime import datetime

# 查询某车次的整条BCU数据
def query_lineData_by_tID(t_id):
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


# 整车次分割成BCU时间片段
def df_slide_window_part(data_df, window_len=80, step=10):
    line_part_list = []
    part_time_list = []
    start = 0
    while (start + window_len < len(data_df)):
        new_part = data_df.iloc[start:start + window_len, :]  # 去窗口中间位置的time属性作为该part的时间戳，用于从LKJ表中获取公里标和经纬度
        line_part_list.append(new_part)
        part_time_list.append(data_df.iloc[start + window_len // 2, 0])
        start += step
    return line_part_list, part_time_list


# 根据t_id，time 查询LKJ表中的 公里标:kilometer
def query_kilo_by_time(time, t_id):
    db = pymysql.connect("localhost", "root", "root123", "shuohuang", charset='utf8')

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    sql = "select time,kilometer from LKJ where tId={} and time >={} limit 1".format(t_id,"\'" + str(time) + "\'")
    cursor.execute(sql)

    # 使用 fetchone() 方法获取单条数据.
    # data=(time,kilometer)
    data = cursor.fetchone()  # 结果返回元组格式，需要转化为list,便于传入DataFrame
    # 关闭数据库连接
    db.close()
    return data

# 根据kilometer 查询对应的经纬度
def query_jwd_by_kilometer(kilometer):
    db = pymysql.connect("localhost", "root", "root123", "shuohuang", charset='utf8')

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    sql = "select l1,l2 from jwd where kilometer = {}".format(kilometer)
    cursor.execute(sql)

    # 使用 fetchone() 方法获取单条数据.
    # data=(time,kilometer)
    data = cursor.fetchone()  # 结果返回元组格式，需要转化为list,便于传入DataFrame
    # 关闭数据库连接
    db.close()
    return data

if __name__ == '__main__':
    # df = query_lineData_by_tID(3)
    # df_slide_window_part(df)
    query_jwd_by_kilometer(4.600)