import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import pymysql

# 先安装此模块： pip install sqlalchemy
from sqlalchemy import create_engine

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows',None)


os.chdir("/Users/arrnos/PycharmProjects/TimeSeriesClassification/")
feature_path = "MovementAAL/dataset"

engine = create_engine("mysql+pymysql://root:root123@localhost:3306/shuohuang?charset=utf8") # 这里一定要写成mysql+pymysql，不要写成mysql+mysqldb

# 将Excel导入（追加）到sql表
# file = pd.read_excel("MovementAAL/bcu_data/bcu_mysql_table/Z3_LKJ.xlsx")
# # file.to_excel("MovementAAL/bcu_data/bcu_mysql_table/Z3_LKJ.xlsx", encoding='utf8', index=None)
# file.to_sql(name = 'bcu_mysql_table',con = engine,if_exists = 'append',index = False,index_label = False)

# 从sql读取表或查询表内容，转换为DataFrame格式，处理后在存放到数据库表中
# jwd = pd.read_sql("select * from jwd",con=engine)
# jwd.to_excel("MovementAAL/bcu_data/bcu_mysql_table/jwd.xlsx", encoding='utf8', index=None)
# if_exists : {'fail', 'replace', 'append'}
# jwd.to_sql(name = 'jwd',con = engine,if_exists = 'append',index = False,index_label = False)

trainLine = pd.read_sql_table(table_name="bcu_mysql_table",con=engine)
trainLine.to_excel("MovementAAL/bcu_data/bcu_mysql_table/bcu_mysql_table.xlsx", encoding='utf8', index=None)