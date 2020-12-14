import itertools
import math
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import sklearn.metrics
from sklearn import preprocessing
import names
import sys
import pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.stats import ttest_rel
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone

alpha = .05

def load():
    data=pd.read_csv("allhyper.data", sep= ',', na_values='?', names=names.Attributes)

    data_X = data.drop('diagnosis',1)
    data_Y = data[['diagnosis']]

    data_Y_np = data_Y.iloc[:,:].values

    #Cut off everything after the dot.
    for field in data_Y_np :
        a_string = field[0]
        split_string = a_string.split(".", 1)
        field[0] = split_string[0]

    data_Y.iloc[:,:] = data_Y_np
    
    return(data_X, data_Y)



def prepareData(data_X, data_y):

    tf = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26]
    
    index_names = data_X[ pd.isnull(data_X['age']) ].index
    data_X.drop(index_names, inplace=True)
    data_y.drop(index_names, inplace=True)
    
    index_names = data_X[ pd.isnull(data_X['sex']) ].index
    data_X.drop(index_names, inplace=True)
    data_y.drop(index_names, inplace=True)
    
    data_X_val = data_X.iloc[:,:].values
    
    for index, line in enumerate(data_X_val):
        for index2, field in enumerate(line):
            if index2 == 1 : # sex
                if field == 'M':
                    data_X_val[index][index2] = 1
                elif field == 'F':
                    data_X_val[index][index2] = 0
            elif index2 in tf: # true/false
                if field == 't':
                    data_X_val[index][index2] = 1
                elif field == 'f':
                    data_X_val[index][index2] = 0
                else:
                    data_X_val[index][index2] = 0.5
            elif index2 == 28: # referral location. Missing value treated as "other".
                if field == 'WEST':
                    data_X_val[index][index2] = 0
                elif field == 'STMW':
                    data_X_val[index][index2] = 0.2
    
                elif field == 'SVHC':
                    data_X_val[index][index2] = 0.4
    
                elif field == 'SVI':
                    data_X_val[index][index2] = 0.6
    
                elif field == 'SVHD':
                    data_X_val[index][index2] = 0.8
    
                else:
                    data_X_val[index][index2] = 1
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=0,)
    for index in (17, 19, 21, 23, 25, 27) :
        temp_vector = imp.fit_transform(data_X_val[:,index].reshape(-1,1))
        try:
            data_X_val[:,index] = temp_vector.ravel()
        except ValueError:
        # In case any NaNs are left (due to not all columns having values), replace those NaNs with 0
            data_X_val[:,index] = [0] * len(data_X_val[:,index])
    
    # scale all continuous data to fit 0 to 1
    min_max_scaler = preprocessing.MinMaxScaler()
    
    for index in (0, 17, 19, 21, 23, 25, 27) :
        temp_vector = data_X_val[:,index].reshape(-1,1)
        data_X_val[:,index] = min_max_scaler.fit_transform(temp_vector).ravel()
    
    data_X.iloc[:,:] = data_X_val

    return data_X, data_y

def ranking(data_X, data_Y, k):
    selector = SelectKBest(score_func=chi2, k=20)
    selector.fit(data_X, data_Y)
    results = selector.scores_
    np.nan_to_num(results, copy=False)
    output = dict(zip(ds_X.columns, results))

    return sorted(output.items(), key=lambda x: x[1], reverse=True)

def print_ranking(ranking):
    for name, val in ranking:
        rounded_val = "{0: .2f}".format(val)
        print(f"{name} : {rounded_val}")

def return_n_ranks(data_x, data_y, n):
    selector = SelectKBest(score_func=chi2, k=n)
    new_data = selector.fit_transform(data_x, data_y)
    return (new_data, data_y)

# Perform classification
def kNNClassify(k, metric, features, training_data, test_data) :
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    training_features = training_data[features]
    training_y = training_data['diagnosis']
    knn.fit(training_features, training_y)
    test_features = test_data[features]
    test_y = test_data['diagnosis']
    classification = knn.predict(test_features)
    print( classification )
    return classification


def calculate_accuracy(data_X, data_Y, metric='euclidean', neigh=5, num_att=5):
    clf = KNeighborsClassifier(n_neighbors=neigh, metric=metric)
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1234)
    scores = []
    for train_index, test_index in rskf.split(data_X, data_Y):
        X_train, X_test = data_X.to_numpy()[train_index], data_X.to_numpy()[test_index]
        y_train, y_test = data_Y.to_numpy()[train_index], data_Y.to_numpy()[test_index]
        clf.fit(X_train, y_train.ravel())
        predict = clf.predict(X_test)
        scores.append(balanced_accuracy_score(y_test, predict))

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))
    print("Number of features:", num_att)
    print("Number of Neighbours:", neigh)
    print("Metric type:", metric, "\n")
    
    return mean_score

def find_optimal_attrs(data_X, data_y, metric, neighb) :
    best_num = 0
    best_val = 0.0
    temp_val = 0.0
    trend = 0
    x = len(data_X.columns)
    for num_of_att in range(1,x) :
        new_x, new_y = return_n_ranks(data_X, data_y, num_of_att)
        n1 = pd.DataFrame(data=new_x)
        n2 = pd.DataFrame(data=new_y)
        temp_val = calculate_accuracy(n1, n2, metric, neighb, num_of_att)
        if temp_val > best_val :
            best_val = temp_val 
            best_num = num_of_att
            if trend < 0 :
                trend = 0
            trend += 1
        elif temp_val < best_val :
            if trend > 0 :
                trend = 0
            trend -= 1
        if trend < -3 : # if the last three increments in the number of features 
            break # have led to a decrease in acuracy, stop adding new features
    return best_val, best_num

def t_student_compare(result1, result2):
    test = ttest_rel(result1, result2)
    T = test.statistic
    p = test.pvalue
    
    return T, p

def t_student_print(T, p, description1, description2):
    if math.isnan(p):
        print (description1 + " vs " + description2 + ": p = nan. Wartości są identyczne")
    elif p > alpha:
        print (description1 + " vs " + description2 + ": Brak istotnych różnic statystycznych")
        return 0;
    elif T > 0 :
        print(description1 + " vs " + description2 + ": Algorytm " + description1 + " jest statystycznie istotnie lepszy.")
        print("T = ", T, " p = ", p)
    else :
        print(description1 + " vs " + description2 + ": Algorytm " + description2 + " jest statystycznie istotnie lepszy.")
        print("T = ", T, " p = ", p)


ds_X, ds_y = load()
ds_X, ds_y = prepareData(ds_X, ds_y)

ranking1 = ranking(ds_X, ds_y, 10)
print_ranking(ranking1)

x, y = zip(*ranking1)

plt.title('Ranking przydatności parametrów do klasyfikacji')
plt.ylabel('Ocena')
plt.xticks(rotation='vertical')
plt.bar(x,y)
for index, value in enumerate(y) :
    plt.text(index-.5, value, str(round(value, 2)))
plt.show()


clfs = {
    'kNN_euclidean1' : KNeighborsClassifier(n_neighbors=1),
    'kNN_euclidean5' : KNeighborsClassifier(n_neighbors=5),
    'kNN_euclidean10' : KNeighborsClassifier(n_neighbors=10),
    'kNN_manhattan1' : KNeighborsClassifier(n_neighbors=1, metric='manhattan'),
    'kNN_manhattan5' : KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
    'kNN_manhattan10' : KNeighborsClassifier(n_neighbors=10, metric='manhattan'),
}

n_splits = 5
n_repeats = 2
length = len(ds_X.columns)
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(clfs), length, n_splits*n_repeats))

for n in range(1, len(ds_X.columns)):
    new_x, new_y = return_n_ranks(ds_X, ds_y, n)
    X = pd.DataFrame(data=new_x)
    y = pd.DataFrame(data=new_y)
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            #print(X)
            #print(y)
            clf = clone(clfs[clf_name])
            clf.fit(X.to_numpy()[train], y.to_numpy()[train].ravel())
            y_pred = clf.predict(X.to_numpy()[test])
            scores[clf_id, n, fold_id] = balanced_accuracy_score(y.to_numpy()[test], y_pred) 

    

np.save('resulttts', scores)



'''
x1 = len(ds_X.columns)
metric_types = ['euclidean', 'manhattan']
knn_number = [1,5,10]


scores = np.recarray()

acc_score = {}

for number_of_att in range (1,x1):
    for x_knn in knn_number:
        for metric in metric_types:
            new_x, new_y = return_n_ranks(ds_X, ds_y, number_of_att)
            n1 = pd.DataFrame(data=new_x)
            n2 = pd.DataFrame(data=new_y)
            scores[number_of_att,x_knn,metric]=calculate_accuracy(n1, n2, metric, x_knn, number_of_att)


for x_knn in [1,5,10]:
    neighb = x_knn
    for metric in metric_types:
        acc_score[str(metric) + str(x_knn)] = find_optimal_attrs(ds_X, ds_y, metric, neighb)

print (acc_score)
'''


# Find best classifier
max_accuracy = max(acc_score, key=acc_score.get)
print("Best accuracy classifier type: ", max_accuracy)

if (max_accuracy == 'manhattan1') :
    clf = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
elif (max_accuracy == 'manhattan5') :
    clf = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
elif (max_accuracy == 'manhattan10') :
    clf = KNeighborsClassifier(n_neighbors=10, metric='manhattan')
elif (max_accuracy == 'euclidean1') :
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
elif (max_accuracy == 'euclidean5') :
    clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
elif (max_accuracy == 'euclidean10') :
    clf = KNeighborsClassifier(n_neighbors=10, metric='euclidean')

best_X, best_y = return_n_ranks(ds_X, ds_y, 'all')
print(best_X)
df_X = pd.DataFrame(data=best_X, columns=names.Attributes[:-1])
df_y = pd.DataFrame(data=ds_y, columns=['diagnosis'])

feature_list = []
for item in range (acc_score[max_accuracy][1]) :
    feature_list.append(x[item])

clf.fit(df_X, df_y)
predict = clf.predict(df_X)

df_merged = df_X
df_merged['diagnosis'] = df_y
df_merged['classification'] = predict

# Confusion matrix
pd.set_option('display.max_rows', None)
le = preprocessing.LabelEncoder()
le.fit(df_y)

classnames = list(le.classes_)

cmtr = confusion_matrix(df_y, df_merged['classification'])
print(cmtr)

plt.imshow(cmtr, interpolation='nearest')
plt.xticks(np.arange(0,len(classnames)), classnames)
plt.yticks(np.arange(0,len(classnames)), classnames)

df_cm = pd.DataFrame(cmtr, index=classnames, columns=classnames)

ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)
ax.set_ylabel('Actual class')
ax.set_xlabel('Predicted class')


plt.show()

################################################
## Statistical analysis ########################
################################################

# Calculate difference between k-NN with different metrics
result1 = acc_score['euclidean1']
result2 = acc_score['manhattan1']
T, p = t_student_compare(result1, result2)
t_student_print(T, p, "euclidean k=1", "manhattan k=1")

result1 = acc_score['euclidean5']
result2 = acc_score['manhattan5']
T, p = t_student_compare(result1, result2)
t_student_print(T, p, "euclidean k=5", "manhattan k=5")

result1 = acc_score['euclidean10']
result2 = acc_score['manhattan10']
T, p = t_student_compare(result1, result2)
t_student_print(T, p, "euclidean k=10", "manhattan k=10")

# Calculate difference between k-NN with k equal to 1, 5 or 10

result1 = acc_score['euclidean1']
result2 = acc_score['euclidean5']
T, p = t_student_compare(result1, result2)
t_student_print(T, p, "euclidean k=1", "euclidean k=5")

result1 = acc_score['euclidean5']
result2 = acc_score['euclidean10']
T, p = t_student_compare(result1, result2)
t_student_print(T, p, "euclidean k=5", "euclidean k=10")

result1 = acc_score['manhattan1']
result2 = acc_score['manhattan5']
T, p = t_student_compare(result1, result2)
t_student_print(T, p, "manhattan k=1", "manhattan k=5")

result1 = acc_score['manhattan5']
result2 = acc_score['manhattan10']
T, p = t_student_compare(result1, result2)
t_student_print(T, p, "manhattan k=5", "euclidean k=10")
