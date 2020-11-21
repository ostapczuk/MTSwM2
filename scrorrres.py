import numpy as np


scores = np.load('resulttts.npy')
print('\nScores:\n', scores)

mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)

from scipy.stats import rankdata
ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("\nRanks:\n", ranks)

mean_ranks = np.mean(ranks, axis=0)
print("\nMean ranks:\n", mean_ranks)