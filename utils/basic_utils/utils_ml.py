from cProfile import label
from typing import Dict
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, GroupKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, accuracy_score
from sklearn import tree, svm, ensemble, linear_model, neural_network, neighbors
import numpy as np
from copy import deepcopy
import pandas as pd

def results_report_sklearn_multiclass(clf, X, y) -> Dict:
    """Results report function that is compatible with sklearn's cross_validate function - multi classification

    Args:
        clf (BaseEstimator): classifier that with .predict() and .predict_proba() API
        X (pd.DataFrame or np.ndarray): input data
        y (list or pd.Series or np.ndarray or ): label

    Returns:
        Dict: results dictionary
    """
    y_pred = clf.predict(X)
    try:
        y_pred_prob = clf.predict_proba(X)
        flag_rocauc = True
    except:
        flag_rocauc = False
    y_labels = sorted(list(set(y)))
    results_dict = classification_report(y_true = y, y_pred = y_pred, output_dict=True, zero_division = 0, labels = y_labels)

    cfmtx = confusion_matrix(y, y_pred, labels=y_labels)
    acc = cfmtx.diagonal()/cfmtx.sum(axis=1)

    if (flag_rocauc):
        rocauc = roc_auc_score(y_true = y, y_score = y_pred_prob, multi_class='ovr')

    results_dict_new = {}
    count = 0
    for k, v in results_dict.items():
        if (type(v) is dict):
            for kk, vv in v.items():
                results_dict_new[f"{kk}#{k}"] = vv
            if (k in y_labels):
                results_dict_new[f"acc#{k}"] = acc[count]
                count += 1
        else:
            results_dict_new[k] = v
    results_dict_new["rocauc"] = rocauc
    return results_dict_new

def results_report_sklearn(clf, X, y, return_confusion_mtx=False) -> Dict:
    """Results report function that is compatible with sklearn's cross_validate function - binary classification

    Args:
        clf (BaseEstimator): classifier that with .predict() and .predict_proba() API
        X (pd.DataFrame or np.ndarray): input data
        y (list or pd.Series or np.ndarray or ): label
        return_confusion_mtx (bool, optional): whether to include confusion matrix in the results. Defaults to False.

    Returns:
        Dict: results dictionary
    """
    results_dict = results_report(y_test = y, y_pred = clf.predict(X), labels = None,
        verbose = False, return_confusion_mtx = return_confusion_mtx)
    if (len(set(y)) == 1):
        results_dict["roc_auc"] = results_dict["balanced_acc"]
    else:
        roc_auc = roc_auc_score(y_true=y, y_score = clf.predict_proba(X)[:,1])
        results_dict["roc_auc"] = roc_auc
    return results_dict

def results_report_sklearn_noprob(clf, X, y, return_confusion_mtx=False) -> Dict:
    """Results report function that is compatible with sklearn's cross_validate function  - binary classification
    Don't have to support .predict_proba() API

    Args:
        clf (BaseEstimator): classifier that with .predict() API
        X (pd.DataFrame or np.ndarray): input data
        y (list or pd.Series or np.ndarray or ): label
        return_confusion_mtx (bool, optional): whether to include confusion matrix in the results. Defaults to False.

    Returns:
        Dict: results dictionary
    """
    results_dict = results_report(y_test = y, y_pred = clf.predict(X), labels = None,
        verbose = False, return_confusion_mtx = return_confusion_mtx)
    return results_dict

def results_report(y_test = None, y_pred = None,
        confusion_mtx = None, verbose=True, labels = [0,1], return_confusion_mtx=False):
    """Report a number of metrics of binary classification results
    
    Parameters
    ----------
    y_test : list or numpy array
        Ground truth list
    y_pred : list or numpy array
        Prediction list, should be the same as y_test
    confusion_mtx : list or numpy 2x2 array, optional, if provided, will not use y_test/y_pred
        Confusion Matrix
    verbose : bool, optional
        Whether show the details
    labels: list, optional
        The value of y
    return_confusion_mtx: bool, optional
        Whether to return confusion_mtx
    
    Returns
    -------
    dictionary
        Results dict
    """
    if (confusion_mtx is None):
        try:
            confusion_mtx = confusion_matrix(y_true = y_test, y_pred = y_pred, labels = [False,True])
        except:
            confusion_mtx = confusion_matrix(y_true = y_test, y_pred = y_pred, labels = labels)
    else:
        confusion_mtx = np.array(confusion_mtx)
    
    tn = confusion_mtx[0][0]
    fp = confusion_mtx[0][1]
    fn = confusion_mtx[1][0]
    tp = confusion_mtx[1][1]
    
    acc, rec, pre, f1 = acc_rec_pre_f1_calc(tp=tp, fp=fp, fn=fn, tn=tn)
    _acc, _rec, _pre, f1_neg = acc_rec_pre_f1_calc(tp=tn, fp=fn, fn=fp, tn=tp)
    p = tp + fn
    n = fp + tn
    ssum = p + n

    sens = rec
    if (n == 0):
        spec = 1
    else:
        spec = tn / n
    balanced_acc = (sens + spec) / 2

    if (((tn + fp) == 0) or ((tn + fn) == 0) or ((tp + fp) == 0) or ((tp + fn) == 0)):
        mcc = 1
    else:
        mcc = (tn * tp - fp * fn) / np.sqrt((tn + fp) * (tn + fn) * (tp + fp) * (tp + fn))

    p_yes = (tn+fp)*(tn+fn)/ (ssum**2)
    p_no = (fn+tp)*(fp+tp)/ (ssum**2)
    pp = p_yes + p_no
    if (pp == 1):
        kappa = 1
    else:
        kappa = (acc - pp) / (1 - pp)

    results = {"acc": acc,
               "balanced_acc": balanced_acc,
               "pre": pre,
               "rec": rec,
               "f1": f1,
               "f1_neg":f1_neg,
               "mcc": mcc,
               "kappa": kappa
               }
    cfmtx = [[tp,fn],[fp,tn]]
    if return_confusion_mtx:
        results.update({"cfmtx": cfmtx})
    if (verbose):
        results_string = \
            "acc:{:.3f},balacc:{:.3f},pre:{:.3f},rec:{:.3f},f1:{:.3f},f1_neg:{:.3f},mcc:{:.3f},kappa:{:.3f}".\
            format(acc, balanced_acc, pre, rec, f1, f1_neg, mcc, kappa) + ",cfmtx:" + str(cfmtx)
        print(results_string)
    return results


def get_clf(clf_type, parameters, direct_param_flag = False):
    """A helper function to get the sklearn classifier. This function can be extended anytime

    Args:
        clf_type (str): classifier type, currently support adaboost, svm, rf, dt, lr, mlp, knn
        parameters (dict): parameter dict, with the necessary param stored as param_name:param_value
        direct_param_flag (bool, optional): Whether to directly passing parameters. Defaults to False.

    Raises:
        Exception: unsupported model type

    Returns:
        BaseEstimator: classifier
    """

    if (direct_param_flag):    
        if (clf_type == "adaboost"):
            clf = ensemble.AdaBoostClassifier(**parameters)
        elif (clf_type == "svm"):
            clf = svm.SVC(**parameters)
        elif (clf_type == "rf"):
            clf = ensemble.RandomForestClassifier(**parameters)
        elif (clf_type == "dt"):
            clf = tree.DecisionTreeClassifier(**parameters)
        elif (clf_type == "lr"):
            clf = linear_model.LogisticRegression(**parameters)
        elif (clf_type == "mlp"):
            clf = neural_network.MLPClassifier(**parameters)
        elif (clf_type == "knn"):
            clf = neighbors.KNeighborsClassifier(**parameters)
        else:
            raise Exception("Sorry. clf_type is not supported.")
    else:
        if (clf_type == "adaboost"):
            if ("max_leaf_nodes" not in parameters):
                clf = ensemble.AdaBoostClassifier(n_estimators = parameters["n_estimators"],
                                              base_estimator = tree.DecisionTreeClassifier(
                                                  max_depth= parameters["max_depth"]),
                                              learning_rate = 1 if "learning_rate" not in parameters else parameters["learning_rate"],
                                              random_state = None if "random_state" not in parameters else parameters["random_state"])
            else:
                clf = ensemble.AdaBoostClassifier(n_estimators = parameters["n_estimators"],
                                              base_estimator = tree.DecisionTreeClassifier(
                                                  max_leaf_nodes = parameters["max_leaf_nodes"]),
                                              learning_rate = 1 if "learning_rate" not in parameters else parameters["learning_rate"],
                                              random_state = None if "random_state" not in parameters else parameters["random_state"])
        elif (clf_type == "svm"):
            clf = svm.SVC(kernel='rbf', C=parameters["C"])
        elif (clf_type == "rf"):
            if ("max_leaf_nodes" not in parameters):
                clf = ensemble.RandomForestClassifier(n_estimators=parameters["n_estimators"],
                    max_depth = parameters["max_depth"], random_state = None if "random_state" not in parameters else parameters["random_state"])
            else:
                clf = ensemble.RandomForestClassifier(n_estimators=parameters["n_estimators"],
                    max_leaf_nodes = parameters["max_leaf_nodes"], random_state = None if "random_state" not in parameters else parameters["random_state"])
        elif (clf_type == "dt"):
            if ("max_leaf_nodes" not in parameters):
                clf = tree.DecisionTreeClassifier(max_depth = parameters["max_depth"])
            else:
                clf = tree.DecisionTreeClassifier(max_leaf_nodes = parameters["max_leaf_nodes"])
        elif (clf_type == "lr"):
            if (parameters["penalty"] == "elasticnet"):
                clf = linear_model.LogisticRegression(penalty = parameters["penalty"], l1_ratio = parameters["l1_ratio"], solver = "saga", C = parameters["C"])
            else:
                clf = linear_model.LogisticRegression(penalty = parameters["penalty"], C = parameters["C"])
        elif (clf_type == "mlp"):
            clf = neural_network.MLPClassifier(hidden_layer_sizes = parameters["hidden_layer_sizes"],
                                activation = "relu",
                                solver = parameters["solver"],
                                learning_rate_init = parameters["learning_rate_init"])

        else:
            raise Exception("Sorry. clf_type is not supported.")

    return clf

def acc_rec_pre_f1_calc(tp, fp, fn, tn):
    p = tp + fn
    n = fp + tn
    ssum = p + n

    acc = (tn + tp) / ssum
    if (p == 0):
        rec = 1
    else:
        rec = tp / p

    if (tp+fp == 0):
        pre = 1
    else:
        pre = tp / (tp+fp)

    if ((rec + pre) == 0):
        f1 = 0
    else:
        f1 = 2 * rec * pre / (rec + pre)

    return acc, rec, pre, f1