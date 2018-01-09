import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from numpy import linalg as la
import re
from PyEMD import EMD, EMD2d, EEMD
from pywt import *
import pywt

data = np.random.randint(0, 3, (10, 3))
x = pd.DataFrame(data)
corr = x.corr().values
corr_fea = []
for i, row in enumerate(corr):
    for j, col in enumerate(row):
        if i < j:
            corr_fea.append(col)

# cov协方差
cov = x.cov().values
cov_fea = []
for i, row in enumerate(cov):
    for j, col in enumerate(row):
        if i <=j:
            cov_fea.append(col)