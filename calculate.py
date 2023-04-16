import os
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate, count, selection_rate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score

np.set_printoptions(precision=4)

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


config_names = ['ml_canzian', 'ml_farhan', 'ml_lu', 'ml_saeb', 'ml_wahle', 'ml_wang', 'ml_chikersal', 'ml_xu_interpretable', 'ml_xu_personalized']

eval_task = 'single'
pred_target = 'dep_weekly'
pred_target = 'dep_endterm'
ds_keys = ['INS-W_1', 'INS-W_2', 'INS-W_3', 'INS-W_4']

folder = os.path.join('./tmp/cross_validate/', config_names[0], eval_task, pred_target, ds_keys[0])
dirs = os.listdir(folder)
prefices = set([dir[:-7] for dir in dirs])
cnt = len(dirs) // len(prefices)
print('cnt', cnt)

metrics = {}
# metrics.update({"accuracy": accuracy_score, "precision": precision_score, "recall": recall_score})
# metrics.update({"f1": f1_score, "roc_auc": roc_auc_score, "balanced_accuracy": balanced_accuracy_score})
# metrics.update({"f1": f1_score, "balanced_accuracy": balanced_accuracy_score})
# metrics.update({"false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate})
metrics.update({"balanced_accuracy": balanced_accuracy_score})

for k, ds_key in enumerate(ds_keys):
    n_cfg = len(config_names)
    axes = {}
    for i, config_name in enumerate(config_names):
        print(config_name, ds_key)
        folder = os.path.join('./tmp/cross_validate/', config_name, eval_task, pred_target, ds_key)
        dirs = os.listdir(folder)
        prefices = set([dir[:-7] for dir in dirs])
        cnt = len(dirs) // len(prefices)
        preds = []
        targs = []
        demos = []

        results = {metric: [] for metric in metrics}
        baccs = []

        for j in range(cnt):
            path_pred = os.path.join(folder, 'y_pred_{:03d}.npy'.format(j + 1))
            path_targ = os.path.join(folder, 'y_targ_{:03d}.csv'.format(j + 1))
            path_demo = os.path.join(folder, 'demographic_test_{:03d}.csv'.format(j + 1))

            pred = np.load(path_pred, allow_pickle=True)
            targ = pd.read_csv(path_targ)["y_raw"]
            demo = pd.read_csv(path_demo).iloc[:, 1:]

            preds.append(pred)
            targs.append(targ)
            demos.append(demo)

            targ = targ.values
            # print('targ', targ)
            # print('pred', pred)
            # get the confusion matrix between pred and targ
            tp = np.sum(np.logical_and(pred == 1, targ == 1))
            fn = np.sum(np.logical_and(pred == 0, targ == 1))
            fp = np.sum(np.logical_and(pred == 1, targ == 0))
            tn = np.sum(np.logical_and(pred == 0, targ == 0))
            p, n = tp + fn, fp + tn
            rec = 1 if p == 0 else tp / p
            spec = 1 if n == 0 else tn / n
            bacc = (rec + spec) / 2
            baccs.append(bacc)

            mf = MetricFrame(metrics=metrics, y_true=targ, y_pred=pred, sensitive_features=np.zeros(demo.shape[0]))
            for metric in metrics:
                results[metric].append(mf.overall[metric])
        
        # print('baccs', baccs)
        print('mean bacc', '{:.04f}'.format(np.mean(baccs)))

        for metric in metrics:
            # print(np.array(results[metric]))
            pass # print(metric, '{:.04f}'.format(np.mean(results[metric])))

        pred = np.concatenate(preds)
        targ = pd.concat(targs)
        demo = pd.concat(demos)
        
        mf = MetricFrame(metrics=metrics, y_true=targ, y_pred=pred, sensitive_features=np.zeros(demo.shape[0]))
        
        folder_output = os.path.join('./tmp/cross_validate_output/', config_name, eval_task, pred_target, ds_key)
        mkdir(os.path.join(folder_output))
        mf.overall.to_csv(os.path.join(folder_output, 'overall.csv'))

        # print('balanced_accuracy: {:.04f}'.format(mf.overall['balanced_accuracy']))
    
    pass # plt.show()