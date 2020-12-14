import numpy as np
from matplotlib import pyplot as plt
import names
from tabulate import tabulate
from scipy.stats import ranksums
from scipy.stats import ttest_ind
from sklearn.neighbors import KNeighborsClassifier


scores = np.load('resulttts.npy')
print('\nScores:\n', scores)

scores = np.load('resulttts.npy')
print('\nScores:\n', scores.shape)



clfs = {
    'kNN_euclidean1' : KNeighborsClassifier(n_neighbors=1),
    'kNN_euclidean5' : KNeighborsClassifier(n_neighbors=5),
    'kNN_euclidean10' : KNeighborsClassifier(n_neighbors=10),
    'kNN_manhattan1' : KNeighborsClassifier(n_neighbors=1, metric='manhattan'),
    'kNN_manhattan5' : KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
    'kNN_manhattan10' : KNeighborsClassifier(n_neighbors=10, metric='manhattan'),
}



alfa = .05
w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

# for k in range 7:
#     for i in range(len(clfs)):
#         for j in range(len(clfs)):
#             w_statistic[i, j], p_value[i, j] = ttest_ind(ranks.T[i], ranks.T[j])
#             print("ranks ",  ranks.T[i], ranks.T[j])]

n_splits = 5
n_repeats = 2

scores1 = np.zeros((len(clfs), n_splits*n_repeats))

x=0
for i in range(11):
    x=i
    print("x=: ", i)
    for i in range(6):
        for j in range(10):
            scores1[i][j] = scores[i,x,j]
            #print(scores[i,x,j], " ")
            




    for i in range(len(clfs)):
        for j in range(len(clfs)):
            w_statistic[i, j], p_value[i, j] = ttest_ind(scores1[i], scores1[j])
    #print("t-statistic:\n", w_statistic, "\n\np-value:\n", p_value)


    headers = list(clfs.keys())
    names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    #print("\t-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    #print("\nAdvantage:\n", advantage_table)


    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)


