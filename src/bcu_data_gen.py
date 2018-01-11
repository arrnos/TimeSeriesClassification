import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from numpy import linalg as la
import re
from datetime import timedelta
from datetime import datetime

os.chdir('/Users/arrnos/PycharmProjects/TimeSeriesClassification/')


def bcu_part_gen(num_sample=None, label=0):
    # bcu 表 字段
    columns = ['时间', '大小闸', 'po1', '均衡风缸', '列车管',
               'po2', '预控', '闸1', '闸2', '总风缸']

    if num_sample is None:
        num_sample = np.random.randint(68, 130)

    # 构造每个特征列

    for i in range(num_sample):

        window_len = np.random.randint(68, 110)

        randnoise = np.random.randint(0, 2, window_len)


        # 1.time
        # 设置随机初始时间
        start_time = datetime(2013, 1, 1, 15, 30, 00, 000)
        end_time = datetime(2017, 2, 1, 15, 30, 00, 000)
        time_delta = int((end_time - start_time).total_seconds())
        init_time = start_time + timedelta(0, np.random.randint(0, time_delta), 0)
        delta = timedelta(0, 5, 0)
        time = [time.strftime('%Y-%m-%d %H:%M:%S') for time in (init_time + delta * i for i in range(window_len))]

        # 2.big_small_brake
        brake_status = ['运转.运转', '中立.运转']

        if label == 0:  # 故障1:全是运转-运转
            seq = [int(x * window_len) for x in [0, 0.2, 0.3, 1]]
            stage1 = [brake_status[0]] * (seq[1] - seq[0])
            stage2 = [brake_status[0]] * (seq[2] - seq[1])
            stage3 = [brake_status[0]] * (seq[3] - seq[2])
            big_small_brake = stage1 + stage2 + stage3
        elif label == 1:  # 故障2：全部是运转-运转
            seq = [int(x * window_len) for x in [0, 0.2, 0.3, 1]]
            stage1 = [brake_status[0]] * (seq[1] - seq[0])
            stage2 = [brake_status[0]] * (seq[2] - seq[1])
            stage3 = [brake_status[0]] * (seq[3] - seq[2])
            big_small_brake = stage1 + stage2 + stage3
        elif label == 2:  # 全部是运转-运转
            seq = [int(x * window_len) for x in [0, 0.2, 0.3, 1]]
            stage1 = [brake_status[0]] * (seq[1] - seq[0])
            stage2 = [brake_status[0]] * (seq[2] - seq[1])
            stage3 = [brake_status[0]] * (seq[3] - seq[2])
            big_small_brake = stage1 + stage2 + stage3
        elif label == 3:  # 全部是运转-运转
            seq = [int(x * window_len) for x in [0, 0.2, 0.3, 1]]
            stage1 = [brake_status[0]] * (seq[1] - seq[0])
            stage2 = [brake_status[0]] * (seq[2] - seq[1])
            stage3 = [brake_status[0]] * (seq[3] - seq[2])
            big_small_brake = stage1 + stage2 + stage3

        # 3.po1 目标值
        po1 = np.random.randint(low=599, high=601, size=window_len).tolist()

        # 4.equil_air_cylinder 均衡风缸
        low_high_value = [(598, 601), (600, 603), (603, 606)]
        equil_air_cylinder = []

        if label == 0:  # 故障1:电空制动控制器在运转位，均衡风缸无压力，列车管定压。
            equil_air_cylinder = [0] * window_len


        elif label == 1:  # 故障2：电空制动控制器在运转位，均衡风缸定压，列车管无压力。
            seq = [int(x * window_len) for x in [0, 0.2, 0.3, 0.9, 1]]
            stage1 = np.random.randint(low=low_high_value[0][0], high=low_high_value[0][1], size=(seq[1] - seq[0]))
            stage2 = np.random.randint(low=low_high_value[1][0], high=low_high_value[1][1], size=(seq[2] - seq[1]))
            stage3 = np.random.randint(low=low_high_value[2][0], high=low_high_value[2][1], size=(seq[3] - seq[2]))
            stage4 = np.random.randint(low=low_high_value[1][0], high=low_high_value[1][1], size=(seq[4] - seq[3]))
            equil_air_cylinder = list(stage1) + list(stage2) + list(stage3) + list(stage4)
        elif label == 2:
            seq = [int(x * window_len) for x in [0, 0.2, 0.3, 1]]
            stage1 = np.random.randint(low=low_high_value[0][0], high=low_high_value[0][1], size=(seq[1] - seq[0]))
            stage2 = np.random.randint(low=low_high_value[1][0], high=low_high_value[1][1], size=(seq[2] - seq[1]))
            stage3 = np.random.randint(low=low_high_value[2][0], high=low_high_value[2][1], size=(seq[3] - seq[2]))
            equil_air_cylinder = list(stage1) + list(stage2) + list(stage3)
        elif label == 3:
            seq = [int(x * window_len) for x in [0, 0.2, 0.3, 1]]
            stage1 = np.random.randint(low=low_high_value[0][0], high=low_high_value[0][1], size=(seq[1] - seq[0]))
            stage2 = np.random.randint(low=low_high_value[1][0], high=low_high_value[1][1], size=(seq[2] - seq[1]))
            stage3 = np.random.randint(low=low_high_value[2][0], high=low_high_value[2][1], size=(seq[3] - seq[2]))
            equil_air_cylinder = list(stage1) + list(stage2) + list(stage3)

        # 加入随机噪声
        equil_air_cylinder = list([x + y for x, y in zip(equil_air_cylinder, randnoise)])

        # 5.train_tube 列车管压力
        train_tube = []
        if label == 0:  # 故障1:电空制动控制器在运转位，均衡风缸无压力，列车管定压。
            low_high_value = [(598, 601), (600, 603), (603, 606)]
            seq = [int(x * window_len) for x in [0, 0.2, 0.3, 1]]
            stage1 = np.random.randint(low=low_high_value[0][0], high=low_high_value[0][1], size=(seq[1] - seq[0]))
            stage2 = np.random.randint(low=low_high_value[1][0], high=low_high_value[1][1], size=(seq[2] - seq[1]))
            stage3 = np.random.randint(low=low_high_value[2][0], high=low_high_value[2][1], size=(seq[3] - seq[2]))
            train_tube = list(stage1) + list(stage2) + list(stage3)
        elif label == 1:  # 故障2：电空制动控制器在运转位，均衡风缸定压，列车管无压力。
            train_tube = [0] * window_len

        elif label == 2:
            low_high_value = [(598, 601), (600, 603), (603, 606)]
            seq = [int(x * window_len) for x in [0, 0.2, 0.3, 1]]
            stage1 = np.random.randint(low=low_high_value[0][0], high=low_high_value[0][1], size=(seq[1] - seq[0]))
            stage2 = np.random.randint(low=low_high_value[1][0], high=low_high_value[1][1], size=(seq[2] - seq[1]))
            stage3 = np.random.randint(low=low_high_value[2][0], high=low_high_value[2][1], size=(seq[3] - seq[2]))
            train_tube = list(stage1) + list(stage2) + list(stage3)
        elif label == 3:
            low_high_value = [(598, 601), (600, 603), (603, 606)]
            seq = [int(x * window_len) for x in [0, 0.2, 0.3, 1]]
            stage1 = np.random.randint(low=low_high_value[0][0], high=low_high_value[0][1], size=(seq[1] - seq[0]))
            stage2 = np.random.randint(low=low_high_value[1][0], high=low_high_value[1][1], size=(seq[2] - seq[1]))
            stage3 = np.random.randint(low=low_high_value[2][0], high=low_high_value[2][1], size=(seq[3] - seq[2]))
            train_tube = list(stage1) + list(stage2) + list(stage3)
        # 加入随机噪声
        train_tube = list([x + y for x, y in zip(train_tube, randnoise)])

        # 6. po2
        value = {'运转.运转': 0, '运转.制动': 314, '中立.制动': 431}
        po2 = [value.get(state, 0) for state in big_small_brake]

        # 加入随机噪声
        po2 = list([x + y for x, y in zip(po2, randnoise)])

        # 7. pre_control
        pre_control = []
        for x in po2:
            y = 0
            if x == 0:
                y = x + np.random.randint(0, 3)
            elif x == 314:
                y = 311
            elif x == 431:
                y = 427
            elif x in [72, 79, 98, 100, 108, 114, 119, 131]:
                y = x - 3
            pre_control.append(y)

        # 8. brake1
        brake1 = []
        for x in po2:
            if x in range(0, 5):
                y = x
            elif x == 314:
                y = 296 + np.random.randint(0, 2)
            elif x == 431:
                y = 412
            else:
                y = 0
            brake1.append(y)

        # 9. brake2
        brake2 = []
        for x in brake1:
            if x == 0:
                y = 0
            else:
                y = x + np.random.randint(-2, 0)
            brake2.append(y)

        # 10. all_air_cylinder
        aac_rule = [830, 825, 819, 815, 811, 807, 803, 799, 795, 791, 787, 783, 779, 784, 789, 794, 799, 804, 809, 814,
                    819,
                    824, 829, 834, 830, 825]
        acc1 = (window_len // len(aac_rule)) * aac_rule
        acc2 = aac_rule[:window_len - len(acc1)]
        all_air_cylinder = acc1 + acc2

        columns = columns
        col_list =  time, big_small_brake, po1, equil_air_cylinder, train_tube, po2, pre_control, brake1, brake2, all_air_cylinder
        data = np.c_[col_list]
        df = pd.DataFrame(data, columns=columns)
        # print([len(x) for x in col_list])
        # print(columns)
        # print(df)
        df.to_excel("MovementAAL/bcu_data/label_{}/Bcu_Exception_Label_{}_sample_{}.xlsx".format(label, label, i),
                  index=None)


if __name__ == '__main__':
    for label in range(0, 2):
        bcu_part_gen(label=label)
