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
faireval_folder = "fairevals"
faireval_folder = "fairevals_is_sensitive"
eval_task = 'single'
pred_target = 'dep_weekly'
pred_target = 'dep_endterm'
ds_keys = ['INS-W_1', 'INS-W_2', 'INS-W_3', 'INS-W_4']

sensitive_attrs = ["race", "gender", "student_1stGen", "student_international", 
                    "generation", "orientation_heterosexual", "parent_edu_father", "parent_edu_mother"]

sensitive_attrs_mapping = {
    "race": "Race",
    "gender": "Gender",
    "student_1stGen": "First Generation Student",
    "student_international": "International Student",
    "college_engineer": "Engineering Major",
    "generation": "Generation Status",
    "orientation_heterosexual": "Heterosexual Orientation",
    "parent_edu_father": "Father's Education Level",
    "parent_edu_mother": "Mother's Education Level"
}

metrics = ["accuracy", "recall", "fpr", "fnr"]
metrics_fullname = ["Accuracy", "Recall", "False positive rate", "False negative rate"]

value2label = {
    "race": {0: "Asian", 1: "White", 2: "Biracial", 3: "Black", 4: "Latinx"},
    "gender": {1: "Male", 2: "Non-male", 3: "Non-male", 4: "Non-male", 6: "Non-male"},
    "orientation_heterosexual": {0: "Non-heterosexual", 1: "Heterosexual"},
    "student_international": {0: "Non-international", 1: "International"},
    "student_1stGen": {0: "Non-first-gen", 1: "First-gen"},
    "parent_edu_mother": {0: "Below bachelor's degree", 1: "Bachelor's degree and above"},
    "parent_edu_father": {0: "Below bachelor's degree", 1: "Bachelor's degree and above"},
    "generation": {0: "Non-immigrant", 1: "Immigrant"},
    "disability": {0: "Non-disabled", 1: "Disabled"}
}

for k, ds_key in enumerate(ds_keys):
    n_cfg = len(config_names)
    axes = {}
    for i, config_name in enumerate(config_names):
        print(config_name, ds_key)
        folder = os.path.join('./tmp/cross_validate/', config_name, eval_task, pred_target, ds_key)
        folder = os.path.join('./tmp/cross_validate_is_sensitive/', config_name, eval_task, pred_target, ds_key)
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
           
            # get the confusion matrix between pred and targ
            tp = np.sum(np.logical_and(pred == 1, targ == 1))
            fn = np.sum(np.logical_and(pred == 0, targ == 1))
            fp = np.sum(np.logical_and(pred == 1, targ == 0))
            tn = np.sum(np.logical_and(pred == 0, targ == 0))
            p, n = tp + fn, fp + tn
            rec = 1 if p == 0 else tp / p
            spec = 1 if n == 0 else tn / n
            bacc = (rec + spec) / 2
            baccs.append(round(bacc,3))

        pred = np.concatenate(preds)
        targ = pd.concat(targs)
        demo = pd.concat(demos)
        print(type(pred))
        print(type(targ))
        print(type(demo))

        pred = pd.Series(pred)
        correct_pred = pred.values == targ.values
        correct_pred = pd.Series(correct_pred)
        print(type(correct_pred))

        pred.replace({True: 1, False: 0},inplace=True)
        targ.replace({True: 1, False: 0},inplace=True)
        correct_pred.replace({True: 1, False: 0},inplace=True)

        # print(pred.index)
        # print(targ.index)
        # print(demo.index)
        # print(correct_pred.index)
        targ.index = pred.index
        demo.index = pred.index
        correct_pred = pd.concat([demo,pred.rename('pred_label'),targ.rename('true_label'),correct_pred.rename('correct_pred')], axis=1)

        metrics = {"accuracy": accuracy_score, 
                    "recall": recall_score, 
                    "fnr": false_negative_rate, 
                    "fpr": false_positive_rate
                }

        fair_evals = {}
        for col in demo.columns:
            faireval = MetricFrame(metrics=metrics, y_true=targ, y_pred=pred, sensitive_features=demo[col])
            fair_evals[col] = faireval
        
        folder_output = os.path.join(faireval_folder, config_name, eval_task, pred_target, ds_key)
        mkdir(os.path.join(folder_output))
        # print(folder_output)
        for col, mf in fair_evals.items():
            mf.by_group.to_csv(os.path.join(folder_output, '{}_by_group.csv'.format(col)))
        mf.overall.append(pd.Series(np.mean(baccs), index=['bal_acc'])).to_csv(os.path.join(folder_output, 'overall.csv'.format(col)))
        correct_pred.to_csv(os.path.join(folder_output, 'correct_prediction.csv'),index=False)

    pass # plt.show()