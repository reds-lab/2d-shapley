# 2D Shapley Implementation with KNN proxy
# Parallel for more permutation handling at a time

import numpy as np
from numpy import random
import time
import pickle
import multiprocessing
import argparse
from pathlib import Path
# import torch


parser = argparse.ArgumentParser()
# add_dataset_model_arguments(parser)

parser.add_argument('--pfrom', type=int, required=True,
                    help='permutation starting from')
parser.add_argument('--pto', type=int, required=True,
                    help='permutation until to exclusively')
arg = parser.parse_args() # args conflict with other argument

print(f"pfrom {arg.pfrom}")
  
print(f"pto {arg.pto}")
    
print("end")

train_set, train_labels, test_set, test_labels = pickle.load(open('breast_cancer_clean.data', 'rb'))
# train_set, train_labels, test_set, test_labels, _, _ = pickle.load(open('breast_cancer_perturb_total_cells_2pc.data', 'rb'))


def knnsv2d(train_data, train_labels, test_data, test_labels, K, perm_num):
    
    np.random.seed(perm_num)
    # T = Number of Test Data
    T = len(test_data)
    # N = Number of Train Data
    N = len(train_labels)
    # M = Number of Features
    M = len(train_data[0])
    # 2D SV Matrix
    sv = np.zeros((M,N)) # We will transpose at the end
    feat_count = np.zeros(M)


#     # For each permutation of features # remove this loop, we will loop each permutation separately
#     for i in range(n_perm):

    # Get a random permutation
    perm = np.random.permutation(M) 
#         print("Perm: ", perm[:20])

    for t in range(T): # We will parallelize this loop in different way (technically not needed here)

        y_test = test_labels[t]

        # Feature squared distances from all train points to the current test point
        feat_distances = np.square(train_data - test_data[t]).T # Transpose for easier access

        # Total feature distances from each train point to the current test point
        tot_distances = np.zeros(N)
        
#         shap_time = time.time()
        # Get whether Train Label equals Test Label
        train_2_test_label = (train_labels == y_test).astype(int)
        
        
        # Case: first feature
        feat = perm[0]
        tot_distances = feat_distances[feat]

        rank_rev = np.argsort(-tot_distances) # square root not needed for ranking
        
        # sv1d is shapley values for a given subset of features (not 1D-shapley discusses in paper)
        sv1d = np.zeros(N)
        train_2_test_label_ranked_rev = train_2_test_label[rank_rev]
        
        cur_label = train_2_test_label_ranked_rev[0]
        cur_val = cur_label / N
        
        all_vals = [cur_val]
    
        # we subtract current label with the next label, to see if there are label changes
        # if 0 next data is same class
        # if 1 next data != test class
        # if -1 next data != test class
        train_2_test_label_ranked_diff = train_2_test_label_ranked_rev[1:] - train_2_test_label_ranked_rev[:-1]
        train_2_test_label_ranked_diff_top_K = train_2_test_label_ranked_diff[-K:]
        for label_diff, i in zip(train_2_test_label_ranked_diff, range(N-1,K,-1)): # this will cut before K
            if label_diff:
                cur_val += label_diff / i 
            all_vals.append(cur_val)
        
        for label_diff in train_2_test_label_ranked_diff_top_K: # For Top K
            if label_diff:
                cur_val += label_diff / K
            all_vals.append(cur_val)
            
        sv1d[rank_rev] = all_vals
        sv[perm[1]] -= sv1d

        # the first feature has count 0, because nothing is added
        feat_count += 1
        feat_count[feat] -= 1


        # Case: second to penultimate feature 
        for p in range(1, M-1):
#                 if p % 50 == 0:
#                     print(p)
            feat = perm[p]
            next_feat = perm[p+1]
            
            tot_distances += feat_distances[feat]# to rank
            rank_rev = np.argsort(-tot_distances) # square root not needed for ranking

            sv1d = np.zeros(N)
            train_2_test_label_ranked_rev = train_2_test_label[rank_rev]

            cur_label = train_2_test_label_ranked_rev[0]
            cur_val = cur_label / N

            all_vals = [cur_val]

            train_2_test_label_ranked_diff = train_2_test_label_ranked_rev[1:] - train_2_test_label_ranked_rev[:-1]
            train_2_test_label_ranked_diff_top_K = train_2_test_label_ranked_diff[-K:]
            for label_diff, i in zip(train_2_test_label_ranked_diff, range(N-1,K,-1)): # this will cut before K
                if label_diff:
                    cur_val += label_diff / i 
                all_vals.append(cur_val)

            for label_diff in train_2_test_label_ranked_diff_top_K: # For Top K
                if label_diff:
                    cur_val += label_diff / K
                all_vals.append(cur_val)

            sv1d[rank_rev] = all_vals
            sv[feat] += sv1d
            sv[next_feat] -= sv1d
        
        # Case: last feature
        feat = perm[M-1]
        tot_distances += feat_distances[feat]# to rank
        rank_rev = np.argsort(-tot_distances) # square root not needed for ranking

        sv1d = np.zeros(N)
        train_2_test_label_ranked_rev = train_2_test_label[rank_rev]

        cur_val = train_2_test_label_ranked_rev[0] / N

        all_vals = [cur_val]

        train_2_test_label_ranked_diff = train_2_test_label_ranked_rev[1:] - train_2_test_label_ranked_rev[:-1]
        train_2_test_label_ranked_diff_top_K = train_2_test_label_ranked_diff[-K:]
        for label_diff, i in zip(train_2_test_label_ranked_diff, range(N-1,K,-1)): # this will cut before K
            if label_diff:
                cur_val += label_diff / i 
            all_vals.append(cur_val)

        for label_diff in train_2_test_label_ranked_diff_top_K: # For Top K
            if label_diff:
                cur_val += label_diff / K
            all_vals.append(cur_val)

        sv1d[rank_rev] = all_vals
        sv[feat] += sv1d
        
    return sv, feat_count
    
    
train_data = train_set
train_labels = train_labels
test_data = test_set
test_labels = test_labels
n_perm = 1
K = 10 # We keep K = 10 by default

# We use all test points, but can reduce if needed
test_points = len(test_set)
cycles = 256 # For memory management - number of processes handled a time, if less space, decrease to an acceptable number

pfrom = arg.pfrom
pto = arg.pto

T = len(test_labels[:test_points])
N = len(train_labels)
M = len(train_set[0])


# Create a folder if not yet
folder_name = "breast_2d_knn_permutations_clean" # for clean dataset
# folder_name = "breast_2d_knn_permutations_2pc"
Path(folder_name).mkdir(parents=True, exist_ok=True)


# # # # For memory management - number of processes handled a time, if less space, change "cycles"
pool = multiprocessing.Pool(cycles)
start_time = time.perf_counter()
for perm_num in range(pfrom, pto):
    sv = np.zeros((M,N))
    feat_count = np.zeros(M)
    for i in range(0,test_points,cycles):
        processes = [pool.apply_async(knnsv2d, args=(train_data, train_labels, test_data[t:t+1], test_labels[t:t+1], K,perm_num)) for t in range(i, min(i+cycles,test_points))]
        for p in range(len(processes)):
            res = processes[p].get()
            sv += res[0]
            feat_count += res[1]
    to_save = [sv, feat_count] 
    pickle.dump(to_save, open(folder_name + "/perm_" + str(perm_num) + ".txt", "wb") )
finish_time = time.perf_counter()
print(f"Program finished in {finish_time-start_time} seconds")




print(sv.shape)
