import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import pymysql
from sqlalchemy import create_engine
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows',None)


os.chdir("/Users/arrnos/PycharmProjects/TimeSeriesClassification/MovementAAL/bcu_data/bcu_mysql_table/站点统一需要更新数据库")

engine = create_engine("mysql+pymysql://root:root123@localhost:3306/shuohuang?charset=utf8") # 这里一定要写成mysql+pymysql，不要写成mysql+mysqldb
# for file in ['Z1','Z2','Z3','J2','J3']:
for file in ['SLL.xlsx']:
    file = pd.read_excel("{}".format(file))
    file.to_sql(name = 'SLL',con = engine,if_exists = 'append',index = False,index_label = False)

# time_head = "2016/11/17 "
# file['time'] = time_head +file['time'].apply(lambda  x :str(x))
# file['time']=file['time'].apply(lambda x: datetime.strptime(x,"%Y/%m/%d %H:%M:%S"))
# s = time_head+"05:50:00"
# de  = datetime.strptime(s,"%Y/%m/%d %H:%M:%S")
# index = file['time']<de
# timed = timedelta(1,0,0)
# file.ix[index,3]+=timed
# print(file['time'])
# file.to_excel("MovementAAL/bcu_data/bcu_mysql_table/J3_LKJ.xlsx", encoding='utf8', index=None)

# # J2_detail = pd.read_excel("MovementAAL/bcu_data/bcu_mysql_table/J2_detail.xlsx")
# # J2_detail.to_sql(name = 'lineData',con = engine,if_exists = 'append',index = False,index_label = False)
# jwd = pd.read_sql("select * from jwd",con=engine)
# jwd['kilometer']=jwd['k'].apply(lambda x:float(x[1:4])+0.001*float(x[5:]))
# jwd['lineCode']=jwd['n']
# jwd.drop(['n'],axis=1,inplace=True)
