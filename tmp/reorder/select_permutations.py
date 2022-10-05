# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:50:28 2017

@author: bbrattol
"""
import argparse
from tqdm import tqdm
import numpy as np
import itertools
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from copy import deepcopy


parser = argparse.ArgumentParser(description='Train network on Imagenet')
parser.add_argument('--patchnum', default=10, type=int, 
        help='Number of patch for permutations')
parser.add_argument('--classes', default=50, type=int, 
                    help='Number of permutations to select')
parser.add_argument('--selection', default='max', type=str, 
        help='Sample selected per iteration based on hamming distance: [max] highest; [mean] average')
args = parser.parse_args()

P_list = []
min_dist_list = []

if __name__ == "__main__":
    outname = 'permutations_hamming_%s_%d'%(args.selection,args.classes)
    
    while True:
        P_hat = np.array(list(itertools.permutations(list(range(args.patchnum)), args.patchnum)))
        n = P_hat.shape[0]
        
        for i in tqdm(range(args.classes)):
            if i==0:
                j = np.random.randint(n)
                P = np.array(P_hat[j]).reshape([1,-1])
            else:
                P = np.concatenate([P,P_hat[j].reshape([1,-1])],axis=0)
            
            P_hat = np.delete(P_hat,j,axis=0)
            D = pairwise_distances(P, P_hat, metric='hamming', n_jobs = 8).mean(axis=0).flatten()
            # D = cdist(P,P_hat, metric='hamming').mean(axis=0).flatten()
            
            if args.selection=='max':
                j = D.argmax()
            else:
                m = int(D.shape[0]/2)
                S = D.argsort()
                j = S[np.random.randint(m-10,m+10)]
            
            # if i%100==0:
                # np.save(outname,P)
        # P_list.append(deepcopy(P))
        D_ = pairwise_distances(P, P, metric='hamming', n_jobs = 4)
        np.fill_diagonal(D_,1)
        min_dist = np.min(D_.min(axis = 0) * args.patchnum)
        # min_dist_list.append(min_dist)
        print(min_dist)
        if (args.classes <= 30):
            if (min_dist > 4):
                break
        elif (args.classes <= 100):
            if (min_dist > 3):
                break
        else:
            break

    np.save(outname,P)
    print('file created --> '+outname)
