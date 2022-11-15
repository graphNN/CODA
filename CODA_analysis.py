# graph-CODA analysis

import numpy as np
import torch

from CODA_utils import load_data

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

A, features, labels, train_mask, val_mask, test_mask, adj = load_data('citeseer')


def curve_(x, y):
    plt.subplot(111)
    plt.grid(linestyle="-")
    plt.plot(x, y[0], label="GCN")
    plt.scatter(x, y[0], s=20)
    plt.plot(x, y[1], label="GAT")
    plt.scatter(x, y[1], s=20)
    plt.plot(x, y[2], label="GRAND")
    plt.scatter(x, y[2], s=20)
    plt.plot(x, y[3], label="CODA-GAT")
    plt.scatter(x, y[3], s=20)
    plt.xlabel('Perturbation Rate (%)')  # Perturbation Rate (%) .. Layers
    plt.ylabel('Accuracy (%)')
    plt.xticks(x)
    plt.yticks(np.arange(20, 80, step=10))  # 60  20
    plt.legend(loc=3, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold', style='normal')  # bold
    plt.savefig('over-smoothing.pdf')
    plt.show()


# over_smoothing
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y0 = [70.80, 70.78, 64.65, 52.1, 46.35, 50.45, 43.45, 37.06, 36.33, 40.38]
y1 = [69.03, 72.50, 68.76, 68.01, 67.22, 66.59, 64.96, 61.32, 43.92, 23.48]
y2 = [74.1, 75.9, 75.0, 74.9, 74.4, 73.9, 74.5, 74.1, 73.5, 74.1]
y3 = [72.88, 77.65, 75.7, 75.7, 75.7, 75.7, 75.7, 75.7, 75.7, 75.7]

# robustness
# x = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
# y0 = [70.3, 69.32, 68.16, 67.02, 65.72, 65.42, 63.08, 64.18, 61.98, 60.10]
# y1 = [72.5, 68.82, 66.72, 66.46, 65.06, 65.18, 63.68, 61.24, 61.54, 60.34]
# y2 = [75.4, 74.2, 73.8, 72.8, 72.9, 72.1, 72.6, 71.5, 71.5, 70.8]
# y3 = [77.24, 75.86, 75.4, 74.5, 74.48, 73.82, 74.28, 73.88, 72.58, 72.96]

y = []
y.append(y0)
y.append(y1)
y.append(y2)
y.append(y3)

curve_(x, y)


def visualize(features, labels):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=800)  # 1000
    low = tsne.fit_transform(features)
    plt.figure(figsize=(8, 8))
    plt.subplot(111)
    plt.scatter(x=low[:, 0], y=low[:, 1], s=20, c=labels, alpha=1., marker='.')  # size = 100
    # ax = plt.subplot(111, projection='3d')
    # ax.scatter(xs=low[:, 0], ys=low[:, 1], zs=low[:, 2], s=20, c=labels, alpha=1., marker='.')
    # plt.savefig('citeseer2.pdf')
    plt.show()

# output = torch.load('./CODA_citeseer.pkl').cpu()
# visualize(features, labels)
# visualize(output, labels)

