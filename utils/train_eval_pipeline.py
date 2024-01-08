import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *

from algorithm.base import DepressionDetectionAlgorithmBase, DepressionDetectionClassifierBase
from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm
from algorithm.ml_chikersal import DepressionDetectionAlgorithm_ML_chikersal
from algorithm.ml_xu_personalized import DepressionDetectionAlgorithm_ML_xu_personalized
from data_loader.data_loader_ml import DatasetDict, DataRepo
from utils.cv_split import judge_corner_cvsplit
from utils import path_definitions
from tensorflow.python.data.ops.dataset_ops import FlatMapDataset

def calc_cv_oneloop(clf: DepressionDetectionClassifierBase, data_repo: DataRepo, random_seed_index: int,
    n_splits:int = 20) -> Dict[str, List[float]]:
    """Perform one around of n-fold cross validate

    Args:
        clf (DepressionDetectionClassifierBase): classififer
        data_repo (DataRepo): data repo for cross validation
        random_seed_index (int): index to control random state
        n_splits (int, optional): number of split. Defaults to 20.

    Returns:
        Dict[str, List[float]]: cross validation results
    """
    repeat_time = -1
    # minimal fold bar in case small data size
    pidnum_min = get_min_count_class(labels=data_repo.y, groups=data_repo.pids)

    while True: 
        repeat_time += 1
        cv = StratifiedGroupKFold(n_splits=min(n_splits, pidnum_min),shuffle=True,random_state=42+random_seed_index+repeat_time*1000)
        if (judge_corner_cvsplit(cv, data_repo)):
            continue
        else:
            break
    return cross_validate(clf, X=data_repo.X, y=data_repo.y, groups=data_repo.pids,
                            cv = cv, n_jobs = 1,
                            scoring = utils_ml.results_report_sklearn, return_train_score=True)
@ray.remote
def calc_cv_oneloop_multithread(clf, data_repo, repeat_num):
    """wrapper function with ray multi-thread computation"""
    return calc_cv_oneloop(clf, data_repo, repeat_num)

def calc_cv_oneloop_singlethread(clf, data_repo, repeat_num):
    """wrapper function with single-thread computation"""
    return calc_cv_oneloop(clf, data_repo, repeat_num)

def single_dataset_model(dataset: DatasetDict, algorithm: DepressionDetectionAlgorithmBase,
                         cv_evaluation:bool = True, multi_thread_flag:bool = True,
                         cv_evaluation_repeat_num:int = 1, verbose:int = 0) -> Tuple[DataRepo, ClassifierMixin, List[Dict[str, float]]]:
    """Model training and evaluation within one dataset

    Args:
        dataset (DatasetDict): dataset to be trained and evaluated
        algorithm (DepressionDetectionModelBase): a depression detection algorithm
        cv_evaluation (bool, optional): whether to do cross-validation evaluation. Defaults to True.
        multi_thread_flag (bool, optional): whether to use multi-thread. Defaults to True.
        cv_evaluation_repeat_num (int, optional): number of cross validation repetition. Defaults to 1.
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Returns:
        Tuple[DataRepo, ClassifierMixin, List[Dict[str, float]]]: (DataRepo, classifier, evalaution results)
    """
    if (verbose>0):
        print("Start data prep...")
    start = time.time()
    data_repo = algorithm.prep_data_repo(dataset)
    end1 = time.time()
    if (verbose>0):
        print("Data prep done. Start model prep...")
    clf = algorithm.prep_model(data_repo)
    end2 = time.time()

    if (verbose>0):
        print("Prep data repo time: ", end1 - start)
        print("Prep model time: ", end2 - end1)
    
    if cv_evaluation:
        if (multi_thread_flag):
            data_repo_id = ray.put(data_repo)
            clf_id = ray.put(clf)
            results_list = ray.get([calc_cv_oneloop_multithread.remote(clf_id, data_repo_id, i) for i in range(cv_evaluation_repeat_num)])
        else:
            results_list = [calc_cv_oneloop_singlethread(clf, data_repo, i) for i in range(cv_evaluation_repeat_num)]
        end3 = time.time()
        if (verbose>0):
            print("CV : ", end3 - end2)
        return data_repo, clf, results_list
    else:
        return data_repo, clf, None

def single_dataset_within_user_model(dataset: DatasetDict, algorithm: DepressionDetectionAlgorithmBase,
                        verbose:int = 0) -> Tuple[DataRepo, ClassifierMixin, List[Dict[str, float]]]:
    """Model training and evaluation within one dataset (with a user's past data in the training set and future data in the testing set)

    Args:
        dataset (DatasetDict): dataset to be trained and evaluated
        algorithm (DepressionDetectionModelBase): a depression detection algorithm
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Returns:
        Tuple[DataRepo, ClassifierMixin, List[Dict[str, float]]]: (DataRepo, classifier, evalaution results)
    """

    if (verbose>0):
        print("Start data prep...")
    start = time.time()
    data_repo = algorithm.prep_data_repo(dataset)
    end1 = time.time()
    if (verbose>0):
        print("Data prep done. Start model prep...")
    clf = algorithm.prep_model(data_repo)
    end2 = time.time()

    if (verbose>0):
        print("Prep data repo time: ", end1 - start)
        print("Prep model time: ", end2 - end1)
    
    if (type(algorithm) is DepressionDetectionAlgorithm_ML_xu_personalized):
        column_name = "pid_origin"
    else:
        column_name = "pid"

    df_data_idx = pd.DataFrame(data_repo.pids, columns=[column_name]).groupby(column_name).apply(lambda x : np.array(x.index))
    df_data_idx = pd.DataFrame(df_data_idx, columns=["idx"])
    df_data_idx["length"] = df_data_idx["idx"].apply(lambda x : len(x))
    df_data_idx['length_test'] = df_data_idx["length"].apply(lambda x : int(np.ceil(x * 0.2)))
    df_data_idx['idx_train'] = df_data_idx.apply(lambda row : row['idx'][:-row["length_test"]], axis = 1)
    df_data_idx['idx_test'] = df_data_idx.apply(lambda row : row['idx'][-row["length_test"]:], axis = 1)
    data_idx_train = np.concatenate(df_data_idx["idx_train"].values)
    data_idx_test = np.concatenate(df_data_idx["idx_test"].values)
    X_train, y_train = data_repo.X.loc[data_idx_train], data_repo.y.loc[data_idx_train]
    X_test, y_test = data_repo.X.loc[data_idx_test], data_repo.y.loc[data_idx_test]

    clf.fit(X = X_train, y = y_train)

    results_train = utils_ml.results_report_sklearn(clf=clf,X=X_train,y=y_train,
        return_confusion_mtx=True)
    results_train = {f"train_{k}":v for k, v in results_train.items()}

    if (verbose > 1):
        print("Train results:", results_train)

    results_test = utils_ml.results_report_sklearn(clf=clf,X=X_test,y=y_test,
        return_confusion_mtx=True)
    results_test = {f"test_{k}":v for k, v in results_test.items()}

    if (verbose > 1):
        print("Test results:", results_test)

    results_full = deepcopy(results_train)
    results_full.update(results_test)

    return data_repo, clf, results_full

def single_dataset_model_dl(dataset: DatasetDict, algorithm: DepressionDetectionAlgorithmBase, 
                         cv_evaluation:bool = True, multi_thread_flag:bool = True,
                         cv_evaluation_repeat_num:int = 1, verbose:int = 0) -> Tuple[DataRepo, ClassifierMixin, List[Dict[str, float]]]:
    """Model training and 5-fold evaluation within one dataset for deep models

    Args:
        dataset (DatasetDict): dataset to be trained and evaluated
        algorithm (DepressionDetectionModelBase): a deep learning depression detection algorithm
        cv_evaluation (bool, optional): whether to do cross-validation evaluation. Defaults to True.
        multi_thread_flag (bool, optional): whether to use multi-thread. Defaults to True.
        cv_evaluation_repeat_num (int, optional): number of cross validation repetition. Defaults to 1.
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Returns:
        Tuple[DataRepo, ClassifierMixin, List[Dict[str, float]]]: (DataRepo, classifier, evalaution results)
    """
    
    with open(os.path.join(path_definitions.DATA_PATH, "additional_user_setup", "split_5fold_pids.json"), "r") as f:
        split_5fold_pids_dict = json.load(f)
    pids_all = deepcopy(split_5fold_pids_dict[dataset.prediction_target][dataset.key]["all"])

    if (verbose>0):
        print("Start data prep...")
    start = time.time()
    data_repo = algorithm.prep_data_repo(dataset)
    end1 = time.time()
    if (verbose>0):
        print("Data prep done. Start model prep...")
    clf = algorithm.prep_model(data_repo)
    end2 = time.time()

    if (verbose>0):
        print("Prep data repo time: ", end1 - start)
        print("Prep model time: ", end2 - end1)
    
    if cv_evaluation:

        results_list = []
        for cv_evaluation_repeat_count in range(cv_evaluation_repeat_num):
            if (cv_evaluation_repeat_count > 0):
                np.random.seed(42 + cv_evaluation_repeat_count)
                np.random.shuffle(pids_all)
                pids_split = np.array_split(pids_all,5)
                for idx_split, pids_split_one in enumerate(pids_split):
                    split_5fold_pids_dict[dataset.prediction_target][dataset.key][str(idx_split + 1)] = sorted(list(pids_split_one))
                with open(path_definitions.DATA_PATH + "/additional_user_setup/split_5fold_pids.json", "w") as f:
                    json.dump(split_5fold_pids_dict, f)

            results_dict_list = []
            for idx_split in range(1,6):
                ds_train = deepcopy(dataset)
                ds_train.eval_task = "single"
                ds_train.flag_split_filter = f"train:{idx_split}"

                ds_test = deepcopy(dataset)
                ds_test.eval_task = "single"
                ds_test.flag_split_filter = f"test:{idx_split}"
                ds_test_dict = {dataset.key: ds_test}

                data_repo_train_tmp, clf_train_tmp, results_dict_tmp = \
                    two_datasets_model(ds_train, ds_test_dict, algorithm, verbose=verbose)
                results_dict_list.append(results_dict_tmp[dataset.key])

            # merge results
            results_dict_merge = {k:[] for k in results_dict_list[0].keys()}
            for d in results_dict_list:
                for k,v in d.items():
                    results_dict_merge[k].append(v)
            results_list.append(results_dict_merge)

        end3 = time.time()
        if (verbose>0):
            print("CV : ", end3 - end2)
        
        return data_repo, clf, results_list
    else:
        return data_repo, clf, None

def single_dataset_within_user_model_dl(dataset: DatasetDict, algorithm: DepressionDetectionAlgorithmBase, 
                        verbose:int = 0) -> Tuple[DataRepo, ClassifierMixin, List[Dict[str, float]]]:
    """Model training and 5-fold evaluation within one dataset for deep models

    Args:
        dataset (DatasetDict): dataset to be trained and evaluated
        algorithm (DepressionDetectionModelBase): a deep learning depression detection algorithm
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Returns:
        Tuple[DataRepo, ClassifierMixin, List[Dict[str, float]]]: (DataRepo, classifier, evalaution results)
    """
    
    if (verbose>0):
        print("Start data prep...")
    start = time.time()
    data_repo = algorithm.prep_data_repo(dataset)
    end1 = time.time()
    if (verbose>0):
        print("Data prep done. Start model prep...")
    clf = algorithm.prep_model(data_repo)
    end2 = time.time()

    if (verbose>0):
        print("Prep data repo time: ", end1 - start)
        print("Prep model time: ", end2 - end1)

    ds_train = deepcopy(dataset)
    ds_train.eval_task = "single_within_user"
    ds_train.flag_single_within_user_split = "train:0.8"

    ds_test = deepcopy(dataset)
    ds_test.eval_task = "single_within_user"
    ds_test.flag_single_within_user_split = "test:0.2"

    ds_test_dict = {}
    ds_test_dict[dataset.key] = ds_test

    data_repo_tmp, clf_tmp, results_dict = \
        two_datasets_model(ds_train, ds_test_dict, algorithm, verbose=verbose)
    return data_repo, clf, results_dict[dataset.key]


def single_dataset_driver(dataset_dict:Dict[str, Dict[str, DatasetDict]], pred_targets:List[str],
                        ds_keys:List[str], algorithm:DepressionDetectionAlgorithmBase,
                        multi_thread_flag:bool=False, cv_evaluation_repeat_num:int=2,
                        flag_return_datarepo = False, flag_return_clf = False,
                        verbose:int=0) -> Dict[str, object]:
    """Driver function to pick a set of datasets to do the evaluation on single dataset

    Args:
        dataset_dict (Dict[str, Dict[str, DatasetDict]]): a dictionary of diction of DatasetDict.
            First level is prediction_target, second level is ds_key
        pred_targets (List[str]): prediction target
        ds_keys (List[str]): a list of ds_keys to be evaluated
        algorithm (DepressionDetectionModelBase): depression detection algorithm
        multi_thread_flag (bool, optional): whether to use multi-thread. Defaults to False.
        cv_evaluation_repeat_num (int, optional): number of cross validation repetition. Defaults to 2.
        flag_return_datarepo (bool, optional): whether to return DataRepo object. Defaults to False.
        flag_return_clf (bool, optional): whether to return DataRepo object. Defaults to False.
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Returns:
        Dict[str, object]: evaluation results dictionary
    """
    data_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    clf_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    results_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    for pred_target, ds_key in itertools.product(pred_targets, ds_keys):
        if (verbose >= 1):
            print("=" * 10, pred_target, ds_key, "=" * 10)
        ds_tmp = deepcopy(dataset_dict[pred_target][ds_key])
        ds_tmp.eval_task = "single"

        if (issubclass(type(algorithm), DepressionDetectionAlgorithm_DL_erm)):
            print("entering DL mode!")
            data_repo_tmp, clf_tmp, results_list_tmp = \
                single_dataset_model_dl(ds_tmp, algorithm,
                cv_evaluation=True, multi_thread_flag=multi_thread_flag, cv_evaluation_repeat_num=1,verbose=verbose)
        else:
            data_repo_tmp, clf_tmp, results_list_tmp = \
                single_dataset_model(ds_tmp, algorithm,
                cv_evaluation=True, multi_thread_flag=multi_thread_flag, cv_evaluation_repeat_num=cv_evaluation_repeat_num,verbose=verbose)
        
        if flag_return_datarepo:
            data_repo_ptds[pred_target][ds_key] = data_repo_tmp
        if flag_return_clf:
            clf_repo_ptds[pred_target][ds_key] = clf_tmp
        results_repo_ptds[pred_target][ds_key] = deepcopy(results_list_tmp)
    single_dataset_dict = {
        "results_repo": results_repo_ptds
    }
    if flag_return_datarepo:
        single_dataset_dict["data_repo"] = data_repo_ptds
    if flag_return_clf:
        single_dataset_dict["clf_repo"] = clf_repo_ptds
    return single_dataset_dict

def single_dataset_within_user_driver(dataset_dict:Dict[str, Dict[str, DatasetDict]], pred_targets:List[str],
                        ds_keys:List[str], algorithm:DepressionDetectionAlgorithmBase,
                        multi_thread_flag:bool=False, cv_evaluation_repeat_num:int=2,
                        flag_return_datarepo = False, flag_return_clf = False,
                        verbose:int=0) -> Dict[str, object]:
    """Driver function to evaluate on one dataset (using previous weeks to predict future weeks)

    Args:
        dataset_dict (Dict[str, Dict[str, DatasetDict]]): a dictionary of diction of DatasetDict.
            First level is prediction_target, second level is ds_key
        pred_targets (List[str]): prediction target
        ds_keys (List[str]): a list of ds_keys to be evaluated
        algorithm (DepressionDetectionModelBase): depression detection algorithm
        multi_thread_flag (bool, optional): whether to use multi-thread. Defaults to False.
        cv_evaluation_repeat_num (int, optional): number of cross validation repetition. Defaults to 2.
        flag_return_datarepo (bool, optional): whether to return DataRepo object. Defaults to False.
        flag_return_clf (bool, optional): whether to return DataRepo object. Defaults to False.
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Returns:
        Dict[str, object]: evaluation results dictionary
    """
    data_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    clf_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    results_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    for pred_target, ds_key in itertools.product(pred_targets, ds_keys):
        if (verbose >= 1):
            print("=" * 10, pred_target, ds_key, "=" * 10)
        ds_tmp = deepcopy(dataset_dict[pred_target][ds_key])
        ds_tmp.eval_task = "single_within_user"

        if (issubclass(type(algorithm), DepressionDetectionAlgorithm_DL_erm)):
            print("entering DL mode!")
            data_repo_tmp, clf_tmp, results_dict_tmp = \
                single_dataset_within_user_model_dl(ds_tmp, algorithm, verbose=verbose)
        else:
            data_repo_tmp, clf_tmp, results_dict_tmp = \
                single_dataset_within_user_model(ds_tmp, algorithm, verbose=verbose)
        
        if flag_return_datarepo:
            data_repo_ptds[pred_target][ds_key] = data_repo_tmp
        if flag_return_clf:
            clf_repo_ptds[pred_target][ds_key] = clf_tmp
        results_repo_ptds[pred_target][ds_key] = deepcopy(results_dict_tmp)
    single_dataset_dict = {
        "results_repo": results_repo_ptds
    }
    if flag_return_datarepo:
        single_dataset_dict["data_repo"] = data_repo_ptds
    if flag_return_clf:
        single_dataset_dict["clf_repo"] = clf_repo_ptds
    return single_dataset_dict

def two_datasets_model(dataset_train: DatasetDict, dataset_test_dict: Dict[str,DatasetDict], algorithm: DepressionDetectionAlgorithmBase,
                        verbose:int = 0) -> Tuple[DataRepo, ClassifierMixin, List[Dict[str, float]]]:
    """Model training and evaluation across different datasets.

    Args:
        dataset_train (DatasetDict): The training set
        dataset_test_dict (Dict[str,DatasetDict]): A dictionary of datasets to be tested on
        algorithm (DepressionDetectionModelBase): Depression detection algorithm
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Raises:
        ValueError: will only occur when training deep models. dataset_train is not properly setup

    Returns:
        Tuple[DataRepo, ClassifierMixin, List[Dict[str, float]]]: (DataRepo, classifier, evalaution results)
    """

    if (verbose>0):
        print("Start data prep...")
    start = time.time()
    data_repo_train = algorithm.prep_data_repo(dataset_train, flag_train=True)
    end1 = time.time()
    if (verbose>0):
        print("Data prep done. Start model prep...")
    clf_train = algorithm.prep_model(data_repo_train)
    clf_train.fit(X=data_repo_train.X,y=data_repo_train.y)
    end2 = time.time()

    if type(data_repo_train.X) is FlatMapDataset:
        data_repo_train_eval = algorithm.prep_data_repo(dataset_train, flag_train=False)
    elif type(data_repo_train.X) is dict:
        if ("val_whole" in data_repo_train.X and type(data_repo_train.X["val_whole"]) is FlatMapDataset):
            data_repo_train_eval = algorithm.prep_data_repo(dataset_train, flag_train=False)
        else:
            raise ValueError
    else:
        data_repo_train_eval = data_repo_train
    results_train = utils_ml.results_report_sklearn(clf=clf_train,X=data_repo_train_eval.X,y=data_repo_train_eval.y,
        return_confusion_mtx=True)
    results_train = {f"train_{k}":v for k, v in results_train.items()}

    if (verbose>0):
        print("Prep data repo time: ", end1 - start)
        print("Prep model time: ", end2 - end1)

    if (verbose > 1):
        print("Train results:", results_train)

    if (type(algorithm) is DepressionDetectionAlgorithm_ML_chikersal or type(algorithm) is DepressionDetectionAlgorithm_ML_xu_personalized):
        print("restarting ray to release memory for chikersal/xu_personalized alg")
        try:
            ray.shutdown()
            ray.init(num_cpus=NJOB, ignore_reinit_error=True)
        except:
            pass

    results_dict = {}
    for ds_key_test, dataset_test in dataset_test_dict.items():
        if (verbose > 0):
            print(f"Test on: {ds_key_test}")
        data_repo_test = algorithm.prep_data_repo(dataset_test, flag_train=False)

        results_test = utils_ml.results_report_sklearn(clf=clf_train,X=data_repo_test.X,y=data_repo_test.y,
            return_confusion_mtx=True)
        results_test = {f"test_{k}":v for k, v in results_test.items()}
        
        if (verbose > 1):
            print("Test results:", results_test)

        results_full = deepcopy(results_train)
        results_full.update(results_test)
        results_dict[ds_key_test] = deepcopy(results_full)

    return data_repo_train, clf_train, results_dict

def two_datasets_driver(dataset_dict:Dict[str, Dict[str, DatasetDict]], pred_targets:List[str],
                        ds_keys:List[str], algorithm:DepressionDetectionAlgorithmBase,
                        flag_return_datarepo = False, flag_return_clf = False,
                        verbose:int=0) -> Dict[str, object]:
    """Driver function to evaluate across two datasets

    Args:
        dataset_dict (Dict[str, Dict[str, DatasetDict]]): a dictionary of diction of DatasetDict.
            First level is prediction_target, second level is ds_key
        pred_targets (List[str]): prediction target
        ds_keys (List[str]): a list of ds_keys to be evaluated
        algorithm (DepressionDetectionModelBase): depression detection algorithm
        flag_return_datarepo (bool, optional): whether to return DataRepo object. Defaults to False.
        flag_return_clf (bool, optional): whether to return DataRepo object. Defaults to False.
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Returns:
        Dict[str, object]: evaluation results dictionary
    """
    data_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    clf_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    results_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    
    for pred_target in pred_targets:
        for ds_key_train in ds_keys:
            if (verbose >= 1):
                print("=" * 10, pred_target, ds_key_train, "=" * 10)
            ds_train = deepcopy(dataset_dict[pred_target][ds_key_train])
            ds_train.eval_task = "two"
            ds_test_dict = {}
            for ds_key_test_tmp in [i for i in ds_keys if i != ds_key_train]:
                ds_test_tmp = deepcopy(dataset_dict[pred_target][ds_key_test_tmp])
                ds_test_tmp.eval_task = "two"
                ds_test_dict[ds_key_test_tmp] = ds_test_tmp

            data_repo_train_tmp, clf_train_tmp, results_dict_tmp = \
                two_datasets_model(ds_train, ds_test_dict, algorithm, verbose=verbose)
            
            if flag_return_datarepo:
                data_repo_ptds[pred_target][ds_key_train] = data_repo_train_tmp
            if flag_return_clf:
                clf_repo_ptds[pred_target][ds_key_train] = clf_train_tmp
            results_repo_ptds[pred_target][ds_key_train] = deepcopy(results_dict_tmp)
            
    two_datasets_dict = {
        "results_repo": results_repo_ptds
    }

    if flag_return_datarepo:
        two_datasets_dict["data_repo"] = data_repo_ptds
    if flag_return_clf:
        two_datasets_dict["clf_repo"] = clf_repo_ptds
    return two_datasets_dict

def two_datasets_overlap_driver(dataset_dict:Dict[str, Dict[str, DatasetDict]], pred_targets:List[str],
                                ds_keys:List[str], algorithm:DepressionDetectionAlgorithmBase,
                                flag_return_datarepo = False, flag_return_clf = False,
                                verbose:int=0) -> Dict[str, object]:
    """Driver function to evaluate across overlapping users in each dataset

    Args:
        dataset_dict (Dict[str, Dict[str, DatasetDict]]): a dictionary of diction of DatasetDict.
            First level is prediction_target, second level is ds_key
        pred_targets (List[str]): prediction target
        ds_keys (List[str]): a list of ds_keys to be evaluated
        algorithm (DepressionDetectionModelBase): depression detection algorithm
        flag_return_datarepo (bool, optional): whether to return DataRepo object. Defaults to False.
        flag_return_clf (bool, optional): whether to return DataRepo object. Defaults to False.
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Returns:
        Dict[str, object]: evaluation results dictionary
    """
    data_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    clf_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    results_repo_ptds = {pred_target:{} for pred_target in pred_targets}

    with open(os.path.join(path_definitions.DATA_PATH, "additional_user_setup", "overlapping_pids.json"), "r") as f:
        overlapping_pids_dict = json.load(f)
    
    for pred_target in pred_targets:
        for ds_key_train in ds_keys:
            institution1, year1 = ds_key_train.split("_")
            ds_key_test_list = []
            for ds_key in ds_keys:
                institution2, year2 = ds_key.split("_")
                if (institution2 == institution1 and year1 != year2):
                    ds_key_test_list.append(ds_key)

            if (verbose >= 1):
                print("=" * 10, pred_target, ds_key_train, "=" * 10)
            ds_train = deepcopy(dataset_dict[pred_target][ds_key_train])
            ds_train.eval_task = "two_overlap"
            # filter 
            ds_train.datapoints = ds_train.datapoints[ds_train.datapoints["pid"].isin(set(overlapping_pids_dict[pred_target][ds_key_train][ds_key_train]))]
            ds_train.flag_overlap_filter = "train:" + ds_key_train

            ds_test_dict = {}
            for ds_key_test in ds_key_test_list:
                ds_test = deepcopy(dataset_dict[pred_target][ds_key_test])
                ds_test.eval_task = "two_overlap"
                ds_test.datapoints = ds_test.datapoints[ds_test.datapoints["pid"].isin(set(overlapping_pids_dict[pred_target][ds_key_train][ds_key_test]))]
                ds_test.flag_overlap_filter = "test:" + ds_key_train
                ds_test_dict[ds_key_test] = ds_test

            data_repo_train_tmp, clf_train_tmp, results_dict_tmp = \
                two_datasets_model(ds_train, ds_test_dict, algorithm, verbose=verbose)
            
            if flag_return_datarepo:
                data_repo_ptds[pred_target][ds_key_train] = data_repo_train_tmp
            if flag_return_clf:
                clf_repo_ptds[pred_target][ds_key_train] = clf_train_tmp
            results_repo_ptds[pred_target][ds_key_train] = deepcopy(results_dict_tmp)
            
    two_datasets_overlap_dict = {
        "results_repo": results_repo_ptds
    }

    if flag_return_datarepo:
        two_datasets_overlap_dict["data_repo"] = data_repo_ptds
    if flag_return_clf:
        two_datasets_overlap_dict["clf_repo"] = clf_repo_ptds
    return two_datasets_overlap_dict

def allbutone_datasets_driver(dataset_dict:Dict[str, Dict[str, DatasetDict]], pred_targets:List[str],
                              ds_keys:List[str], algorithm:DepressionDetectionAlgorithmBase,
                              flag_return_datarepo = False, flag_return_clf = False,
                              verbose:int=0) -> Dict[str, object]:
    """Driver function to evaluate model with the leave-one-dataset-out setup

    Args:
        dataset_dict (Dict[str, Dict[str, DatasetDict]]): a dictionary of diction of DatasetDict.
            First level is prediction_target, second level is ds_key
        pred_targets (List[str]): prediction target
        ds_keys (List[str]): a list of ds_keys to be evaluated
        algorithm (DepressionDetectionModelBase): depression detection algorithm
        flag_return_datarepo (bool, optional): whether to return DataRepo object. Defaults to False.
        flag_return_clf (bool, optional): whether to return DataRepo object. Defaults to False.
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Returns:
        Dict[str, object]: evaluation results dictionary
    """
    data_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    clf_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    results_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    
    for pred_target in pred_targets:
        for ds_key_test in ds_keys:
            if (verbose >= 1):
                print("=" * 10, pred_target, ds_key_test, "=" * 10)
            ds_test = deepcopy(dataset_dict[pred_target][ds_key_test])
            ds_test.eval_task = "allbutone"
            
            ds_train_keys = np.sort([i for i in ds_keys if i != ds_key_test])
            ds_train_key = ":".join(ds_train_keys)
            
            ds_train_datapoints_list = [dataset_dict[pred_target][ds_key_train].datapoints for ds_key_train in ds_train_keys]
            ds_train_datapoints = pd.concat(ds_train_datapoints_list).reset_index(drop=True)
            ds_train = DatasetDict(key = ds_train_key, prediction_target=pred_target, datapoints=ds_train_datapoints)
            ds_train.eval_task = "allbutone"
            
            ds_test_dict = {ds_key_test: ds_test}
            data_repo_train_tmp, clf_train_tmp, results_dict_tmp = \
                two_datasets_model(ds_train, ds_test_dict, algorithm, verbose=verbose)
            
            if flag_return_datarepo:
                data_repo_ptds[pred_target][ds_key_test] = data_repo_train_tmp
            if flag_return_clf:
                clf_repo_ptds[pred_target][ds_key_test] = clf_train_tmp
            results_repo_ptds[pred_target][ds_key_test] = deepcopy(results_dict_tmp[ds_key_test])
            if (verbose > 1):
                print("-" * 5, "\n", results_dict_tmp[ds_key_test], "\n", "-" * 5)

    allbutone_datasets_dict = {
        "results_repo": results_repo_ptds
    }

    if flag_return_datarepo:
        allbutone_datasets_dict["data_repo"] = data_repo_ptds
    if flag_return_clf:
        allbutone_datasets_dict["clf_repo"] = clf_repo_ptds
    return allbutone_datasets_dict

def crossgroup_datasets_driver(dataset_dict:Dict[str, Dict[str, DatasetDict]], pred_targets:List[str],
                               ds_keys:List[str], algorithm:DepressionDetectionAlgorithmBase,
                               flag_return_datarepo = False, flag_return_clf = False,
                               verbose:int=0) -> Dict[str, object]:
    """Driver function to evaluate model with the cross-insitute and cross-year setup

    Args:
        dataset_dict (Dict[str, Dict[str, DatasetDict]]): a dictionary of diction of DatasetDict.
            First level is prediction_target, second level is ds_key
        pred_targets (List[str]): prediction target
        ds_keys (List[str]): a list of ds_keys to be evaluated
        algorithm (DepressionDetectionModelBase): depression detection algorithm
        flag_return_datarepo (bool, optional): whether to return DataRepo object. Defaults to False.
        flag_return_clf (bool, optional): whether to return DataRepo object. Defaults to False.
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Returns:
        Dict[str, object]: evaluation results dictionary
    """
    data_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    clf_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    results_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    
    for pred_target in pred_targets:
        for ds_key_test_list in itertools.combinations(ds_keys,2):
            # skip inappropriate groups
            institution1, year1 = ds_key_test_list[0].split("_")
            institution2, year2 = ds_key_test_list[1].split("_")
            if (institution2 != institution1) and (year1 != year2):
                continue

            ds_train_keys = np.sort([i for i in ds_keys if i not in ds_key_test_list])
            ds_test_keys = np.sort([i for i in ds_keys if i in ds_key_test_list])
            ds_train_key = ":".join(ds_train_keys)
            ds_test_key = ":".join(ds_test_keys)

            if (verbose >= 1):
                print("=" * 10, pred_target, ds_test_key, "=" * 10)
            
            ds_train_datapoints_list = [dataset_dict[pred_target][ds_key_train].datapoints for ds_key_train in ds_train_keys]
            ds_train_datapoints = pd.concat(ds_train_datapoints_list).reset_index(drop=True)
            ds_train = DatasetDict(key = ds_train_key, prediction_target=pred_target, datapoints=ds_train_datapoints)
            ds_train.eval_task = "crossgroup"

            ds_test_datapoints_list = [dataset_dict[pred_target][ds_key_test].datapoints for ds_key_test in ds_test_keys]
            ds_test_datapoints = pd.concat(ds_test_datapoints_list).reset_index(drop=True)
            ds_test = DatasetDict(key = ds_test_key, prediction_target=pred_target, datapoints=ds_test_datapoints)
            ds_test.eval_task = "crossgroup"

            ds_test_dict = {ds_test_key: ds_test}
            data_repo_train_tmp, clf_train_tmp, results_dict_tmp = \
                two_datasets_model(ds_train, ds_test_dict, algorithm, verbose=verbose)
            
            if flag_return_datarepo:
                data_repo_ptds[pred_target][ds_test_key] = data_repo_train_tmp
            if flag_return_clf:
                clf_repo_ptds[pred_target][ds_test_key] = clf_train_tmp
            results_repo_ptds[pred_target][ds_test_key] = deepcopy(results_dict_tmp[ds_test_key])
            if (verbose > 1):
                print("-" * 5, "\n", results_dict_tmp[ds_test_key], "\n", "-" * 5)
    crossgroup_datasets_dict = {
        "results_repo": results_repo_ptds
    }
    if flag_return_datarepo:
        crossgroup_datasets_dict["data_repo"] = data_repo_ptds
    if flag_return_clf:
        crossgroup_datasets_dict["clf_repo"] = clf_repo_ptds
    return crossgroup_datasets_dict

def crosscovid_datasets_driver(dataset_dict:Dict[str, Dict[str, DatasetDict]], pred_targets:List[str],
                               ds_keys:List[str], algorithm:DepressionDetectionAlgorithmBase,
                               flag_return_datarepo = False, flag_return_clf = False,
                               verbose:int=0) -> Dict[str, object]:
    """Driver function to evaluate model with the pre/post COVID years setup (only for certain institutes)

    Args:
        dataset_dict (Dict[str, Dict[str, DatasetDict]]): a dictionary of diction of DatasetDict.
            First level is prediction_target, second level is ds_key
        pred_targets (List[str]): prediction target
        ds_keys (List[str]): a list of ds_keys to be evaluated
        algorithm (DepressionDetectionModelBase): depression detection algorithm
        flag_return_datarepo (bool, optional): whether to return DataRepo object. Defaults to False.
        flag_return_clf (bool, optional): whether to return DataRepo object. Defaults to False.
        verbose (int, optional): whether to print intermediate outcome. Defaults to 0.

    Returns:
        Dict[str, object]: evaluation results dictionary
    """
    data_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    clf_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    results_repo_ptds = {pred_target:{} for pred_target in pred_targets}
    
    for pred_target in pred_targets:
        for ds_key_test_list in itertools.combinations(ds_keys,2):
            # skip inappropriate groups
            institution1, year1 = ds_key_test_list[0].split("_")
            institution2, year2 = ds_key_test_list[1].split("_")
            # only year [1,2] or year [3,4]
            if not ((int(year1) < 2.5 and int(year2) < 2.5) or (int(year1) > 2.5 and int(year2) > 2.5)):
                continue

            ds_train_keys = np.sort([i for i in ds_keys if i not in ds_key_test_list])
            ds_test_keys = np.sort([i for i in ds_keys if i in ds_key_test_list])
            ds_train_key = ":".join(ds_train_keys)
            ds_test_key = ":".join(ds_test_keys)

            if (verbose >= 1):
                print("=" * 10, pred_target, ds_test_key, "=" * 10)
            
            ds_train_datapoints_list = [dataset_dict[pred_target][ds_key_train].datapoints for ds_key_train in ds_train_keys]
            ds_train_datapoints = pd.concat(ds_train_datapoints_list).reset_index(drop=True)
            ds_train = DatasetDict(key = ds_train_key, prediction_target=pred_target, datapoints=ds_train_datapoints)
            ds_train.eval_task = "crosscovid"

            ds_test_datapoints_list = [dataset_dict[pred_target][ds_key_test].datapoints for ds_key_test in ds_test_keys]
            ds_test_datapoints = pd.concat(ds_test_datapoints_list).reset_index(drop=True)
            ds_test = DatasetDict(key = ds_test_key, prediction_target=pred_target, datapoints=ds_test_datapoints)
            ds_test.eval_task = "crosscovid"

            ds_test_dict = {ds_test_key: ds_test}
            data_repo_train_tmp, clf_train_tmp, results_dict_tmp = \
                two_datasets_model(ds_train, ds_test_dict, algorithm, verbose=verbose)
            
            if flag_return_datarepo:
                data_repo_ptds[pred_target][ds_test_key] = data_repo_train_tmp
            if flag_return_clf:
                clf_repo_ptds[pred_target][ds_test_key] = clf_train_tmp
            results_repo_ptds[pred_target][ds_test_key] = deepcopy(results_dict_tmp[ds_test_key])
            if (verbose > 1):
                print("-" * 5, "\n", results_dict_tmp[ds_test_key], "\n", "-" * 5)
    crosscovid_datasets_dict = {
        "results_repo": results_repo_ptds
    }
    if flag_return_datarepo:
        crosscovid_datasets_dict["data_repo"] = data_repo_ptds
    if flag_return_clf:
        crosscovid_datasets_dict["clf_repo"] = clf_repo_ptds
    return crosscovid_datasets_dict
    
