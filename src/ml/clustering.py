#!/usr/bin/env python
from sklearn.cluster import KMeans

def kmeans(features, k=10):
    km = KMeans(n_clusters=k, init='k-means++', n_init=1, verbose=True)
    km.fit(features)
    return km.labels_

def plot(features, labels):
    import matplotlib.pyplot as plt

    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap=plt.cm.jet)
    plt.show()


if __name__ == '__main__':
    import numpy as np

    features = np.random.rand(512, 2)

    k = 10
    labels = kmeans(features, k=k)

    plot(features, labels)
