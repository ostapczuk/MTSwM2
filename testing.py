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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

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

def cross_validation(metric, neigh, data_X, data_Y, num_att):
    #data2 = pd.DataFrame.join(data_X, data_Y)
    clf = KNeighborsClassifier(n_neighbors=neigh, metric=metric)
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1234)
    scores = []
    for train_index, test_index in rskf.split(data_X, data_Y):
        X_train, X_test = data_X.to_numpy()[train_index], data_X.to_numpy()[test_index]
        y_train, y_test = data_Y.to_numpy()[train_index], data_Y.to_numpy()[test_index]
        clf.fit(X_train, y_train.ravel())
        predict = clf.predict(X_test)
        scores.append(accuracy_score(y_test, predict))

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))
    print("Number of features:", num_att)
    print("Number of Neighbours:", neigh)
    print("Metric type:", metric)

    return mean_score

'''
# Returns 
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
'''


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
'''
ds_train_1, ds_test_1 = cross_validation(ds_X, ds_y, 9)

feature_list = []
for item in range (10) :
    feature_list.append(x[item])

ds_class_1 = kNNClassify(5, 'euclidean', feature_list, ds_train_1, ds_test_1)

ds_merged_1 = ds_test_1
ds_merged_1['classification'] = ds_class_1

ds_merged_1.to_csv("data.csv")

accuracy = sklearn.metrics.accuracy_score(ds_merged_1['diagnosis'], ds_merged_1['classification'])

# Making a dictionary out of class names that will hold results.

#for record in ds_merged_1[ ['diagnosis', 'classification'] ] :
#    predictions[ record['diagnosis'] ][0] += 1
#    if record['diagnosis'] == record['classification'] :
#        predictions[ record['diagnosis'] ][1] += 1

print(accuracy)
'''
print(ds_X)
print (ds_y)
#cross_validation(ds_X, ds_y)

#print(np.unique(ds_y, return_counts=True))
#accuracy, iterations = experiment(ds_X, ds_y)
#print("Number of iterations" , iterations) 
#print("Accuracy score: %.3f" % accuracy)

metric_types = ['euclidean', 'manhattan']

x = len(ds_X.columns)


for num_of_att in range(1,x):
    new_x, new_y = return_n_ranks(ds_X, ds_y, num_of_att)
    n1 = pd.DataFrame(data=new_x)
    n2 = pd.DataFrame(data=new_y)
    for x_knn in [1,5,10]:
        for metric in metric_types:
            neighb = x_knn
            cross_validation(metric, neighb, n1, n2, num_of_att)


