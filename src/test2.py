import numpy as np
from random import randrange

from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize

(features, labels) = make_blobs(n_samples=500, n_features=10, centers=4)
features = normalize(X=features, norm='l2', axis=0)
print(labels.shape)
labels = map(int, labels)
print(labels)
