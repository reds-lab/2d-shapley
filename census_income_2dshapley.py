import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from numpy import random
from math import factorial
import time
from sklearn.datasets import load_iris
from itertools import product
import pickle

import multiprocessing
import argparse

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import os
import numpy as np
from copy import deepcopy

from pathlib import Path


parser = argparse.ArgumentParser()
# add_dataset_model_arguments(parser)

# parser.add_argument('--cnum', type=int, required=True,
#                     help='number of cuda in the server')
parser.add_argument('--procs', type=int, required=True,
                    help='number of processors')
parser.add_argument('--pfrom', type=int, required=True,
                    help='permutation starting from')
parser.add_argument('--pto', type=int, required=True,
                    help='permutation until to')
arg = parser.parse_args() # args conflict with other argument

# print(f"args cnum {arg.cnum}")

print(f"procs cnum {arg.procs}")
  
print(f"pfrom {arg.pfrom}")
  
print(f"pto {arg.pto}")
    
values_folder = "census_2d_vals/"
data_folder = "data/"

Path(data_folder).mkdir(parents=True, exist_ok=True)
    
print("end")


train_path = Path(data_folder + 'adult_train.csv')
test_path = Path(data_folder + 'adult_test.csv')
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

test = test[1:]
test.dropna(how="all",inplace=True)
le= LabelEncoder()

for col in train.columns:
    if train[col].dtypes=='object':
        train[col]= le.fit_transform(train[col])

for col in test.columns:
    if test[col].dtypes=='object':
        test[col]= le.fit_transform(test[col])
        
train = train.drop(columns=['fnlwgt'])
test = test.drop(columns=['fnlwgt'])
        
train = train.rename(columns={'Age': 0, 
    'Workclass': 1, 
    'Education': 2, 
    'Education_Num': 3, 
    'Martial_Status': 4, 
    'Occupation': 5, 
    'Relationship': 6, 
    'Race': 7, 
    'Sex': 8, 
    'Capital_Gain': 9,
    'Capital_Loss': 10, 
    'Hours_per_week': 11,
    'Country': 12,
    'Target': 13})
        
        
test = test.rename(columns={'Age': 0, 
    'Workclass': 1, 
    'Education': 2, 
    'Education_Num': 3, 
    'Martial_Status': 4, 
    'Occupation': 5, 
    'Relationship': 6, 
    'Race': 7, 
    'Sex': 8, 
    'Capital_Gain': 9,
    'Capital_Loss': 10, 
    'Hours_per_week': 11,
    'Country': 12,
    'Target': 13})


categories = ['Age', 'Workclass', 'Education','Education_Num','Martial_Status', 'Occupation','Relationship','Race','Sex','Capital_Gain','Capital_Loss','Hours_per_week','Country','Target']

train_len = len(train)
feat_len = train.shape[1]-1

nums = 10


def calc_perf(pos, feat_len, train, test, data_perm, feat_perm):
    #print(pos)
    if pos % 10000 == 0:
        print(pos)
    i = int(pos / feat_len)
    j = int(pos % feat_len)
    data_i = data_perm[i] # curr data \n",
    subset_data_i = data_perm[:i+1] # data indices including i\n",

    # get data including i\n",
    sub_train_i = deepcopy(train).iloc[subset_data_i,:]

    feat_j = feat_perm[j] # curr feature\n",
    # feature indices including j (= removing features after j)\n",
    subset_feat_j = feat_perm[:j+1] 

    ## i and j\n",
    sub_train_i_j = deepcopy(sub_train_i).iloc[:,subset_feat_j] # we do modify  \n",

    DC = DecisionTreeClassifier()
    
    acc_i_j = 0
    for i in range(nums):
        DC.fit(sub_train_i_j, sub_train_i.iloc[:,feat_len]) # sub_train_i.iloc[:,feat_len] - labels
        pred = DC.predict(test.iloc[:,subset_feat_j])
        acc_i_j += accuracy_score(test.iloc[:,feat_len],pred)
    acc_i_j /= nums
    return acc_i_j



train_arr = np.arange(len(train))
feat_arr = np.arange(train.shape[1]-1)

    
# values of cells in the matrix, initialized to 0\n",
# will keep a 2D array for faster changes rather than a 2D dictionary\n",
cells = np.zeros((len(train_arr), len(feat_arr)))

verbose = False
# perm_num = 15
pfrom = arg.pfrom
print(f"p from {pfrom}")
pto = arg.pto
print(f"p to {pto}")

perms = range(pfrom, pto)
print(f"perms: {perms}")

for p in perms:


#     if p % 100000 == 0:
    print("p: ", p)
    # get a data permutation\n",

    data_perm_time = time.time()
    data_perm = random.permutation(train_arr)
    data_perm_time = time.time() - data_perm_time

    # get a feature permutation\n",
    feat_perm_time = time.time()
    feat_perm = random.permutation(feat_arr)
    feat_perm_time = time.time() - feat_perm_time

    # for each cell in matrix, we measure update values (just from top to bottom is fine)\n",
    # when not considering a certain feature, we change value to -1, (since 0 is still related to some value)\n",
    # this is straighforward\n",

    # example will remove later\n",
    # a b c d ...\n",
    # e f g h ...\n",
    # i j k l ...\n",

    # we go through row by row\n",
    # we need to save calculations from last row, so that we don't retrain the model (faster)\n",

    # a - calc (0,0)\n",
    # b - calc (0,0) (reuse) + calc (0,1) \n",
    # c - calc (0,1) (reuse) + calc (0,2)\n",
    # d - calc (0,2) (reuse) + calc (0,3)\n",
    # ...\n",
    # e - calc (1,0)\n",

    # will need to calculate all cells each one time (retraining) \n",
    # then just resuing all values\n",

    # make a map/array that keeps 3 previous calculations\n",

    # we instead have an array/map that keeps all m (all features) previous calculations \n",
    # and we have one more variable that keeps the previous calculation (except for the last column)\n",

    # key is only category\n",
    # we only save data = the number of categories, no need for older values, we don't use them\n",
    saved_values = np.zeros(feat_len)
    last_value = 0


    pool = multiprocessing.Pool(arg.procs)
    start_time = time.perf_counter()
    processes = [pool.apply_async(calc_perf, args=(pos, feat_len, train, test, data_perm, feat_perm,)) for pos in range(0, train_len * feat_len)]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")

    model_perfs = np.zeros((train_len, feat_len))
    for i in range(len(result)):
        model_perfs[int(i/feat_len)][i%feat_len] = result[i]
    vals_1 = np.zeros((train_len, feat_len))
    
    for position in range(len(result)):

        i = int(position / feat_len)
        j = int(position % feat_len)

        # val00 val01
        # val10 val11:cur_val

        val00 = 0
        val01 = 0
        val10 = 0
        if i > 0:
            val01 = model_perfs[i-1][j]
            if j > 0:
                val00 = model_perfs[i-1][j-1]
        if j > 0:
            val10 = model_perfs[i][j-1]

        val11 = model_perfs[i][j]

        cur_perf = model_perfs[i][j] + val00 - val01 - val10

        data_i = data_perm[i] # curr data 
        feat_j = feat_perm[j] # curr feature

        vals_1[data_i][feat_j] = cur_perf
    
    to_save = [vals_1]
    pickle.dump(to_save, open(values_folder + "census_2d_values_permutation_" + str(p) + ".txt", "wb") )
    
print("finished ", p, " permutations")
