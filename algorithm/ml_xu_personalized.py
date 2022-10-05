"""
Implementation of prior depression detection algorithm:

Xuhai Xu, Prerna Chikersal, Janine M. Dutcher, Yasaman S. Sefidgar, Woosuk Seo, Michael J. Tumminia,
Daniella K. Villalba, Sheldon Cohen, Kasey G. Creswell, J. David Creswell,
Afsaneh Doryab, Paula S. Nurius, Eve Riskin, Anind K. Dey, and Jennifer Mankoff. 2021.
Leveraging Collaborative-Filtering for Personalized Behavior Modeling: A Case Study of Depression Detection among College Students.
Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 5, 1 (March 2021), 1–27.
https: //doi.org/10.1145/3448107
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.ml_basic import DepressionDetectionAlgorithm_ML_basic
from data_loader.data_loader_ml import DatasetDict, DataRepo
from algorithm.base import DepressionDetectionClassifierBase
from sklearn.metrics import pairwise_distances
import signal

def np_take(U,kx,ky):
    return np.take(np.take(U,kx, axis=0), ky,axis=1)

class DepressionDetectionClassifier_ML_xu_personalized(DepressionDetectionClassifierBase):
    """ Classifier for Xu et al. personalized work.
        Save the training data to use a memory-based collaborative filtering model """

    def __init__(self, selected_feature_th_dict):
        self.selected_feature_th_dict = selected_feature_th_dict
        self.selected_features = list(selected_feature_th_dict.keys())
    
    def fit(self, X, y):
        assert set(self.selected_features).issubset(set(X.columns))
        set_random_seed(42)
        self.data_memory_X = deepcopy(X)
        self.data_memory_y = deepcopy(y)
        return None
    
    def calc_th(self):
        self.th_dict = {}
        
        for selected_feature in self.selected_features:
            df_userbehavior_target = np.array(self.data_memory_X[selected_feature].to_list())
            sim_cor_square = (1 - pairwise_distances(df_userbehavior_target, metric = "correlation")).astype(np.float64)
            sim_cor_square = sim_cor_square * sim_cor_square

            sim_mtx_buf = deepcopy(sim_cor_square)    
            np.fill_diagonal(sim_mtx_buf, 0)
            sim_mtx_buf_rowsum = sim_mtx_buf.sum(axis = 1)
            train_score = ((np.matmul(sim_mtx_buf, self.data_memory_y)) / (sim_mtx_buf_rowsum + 0.000001))

            train_score1 = train_score[np.where(self.data_memory_y)[0]]
            train_score2 = train_score[np.where(1 - self.data_memory_y)[0]]

            mean1 = np.mean(train_score1)
            mean2 = np.mean(train_score2)

            self.th_dict[selected_feature] = (mean1 + mean2) / 2
    
    def collaborative_filtering_pred(self, sim_mtx, Y_memory, score_th): 
        sim_weight = deepcopy(sim_mtx)
        sim_weight_abs = np.abs(sim_weight)
        zeroout_idx = np.where(sim_weight_abs < np.percentile(sim_weight_abs, 25))
        sim_weight[zeroout_idx] = 0
        pred_score = np.matmul(sim_weight, Y_memory)
        pred_score /= (sim_weight.sum(axis=1) + 0.000001)
        y_pred = pred_score > score_th
        return y_pred
    
    def get_final_majority_voting(self, df_seperate_predictions, div=2):
        th = len(df_seperate_predictions) // div
        predictions = np.array(df_seperate_predictions).sum(axis = 0) > th
        return predictions

    def get_final_majority_voting_prob(self, df_seperate_predictions, div=2):
        th = len(df_seperate_predictions) // div
        probs_repo = np.concatenate([np.linspace(0, 0.5, th+1,endpoint=True),
            np.linspace(0.5,1, len(df_seperate_predictions)-th+1, endpoint=True)[1:]])
        voting_num = np.array(df_seperate_predictions).sum(axis = 0)
        predictions_prob = np.array([[1-probs_repo[int(num)], probs_repo[int(num)]] for num in voting_num])
        return predictions_prob

    def predict(self, X, y=None):
        df_singlefeature_pred = []
        assert set(self.selected_features).issubset(set(X.columns))
        for selected_feature in self.selected_features:
            wk_e, feat = selected_feature.split("::")
            df_userbehavior_memory = np.array(self.data_memory_X[selected_feature].to_list())
            df_userbehavior_target = np.array(X[selected_feature].to_list())
            sim_cor_square = (1 - pairwise_distances(X = df_userbehavior_target,
                                            Y = df_userbehavior_memory, metric = "correlation")).astype(np.float64)
            sim_cor_square = sim_cor_square * sim_cor_square

            y_pred = self.collaborative_filtering_pred(sim_cor_square, self.data_memory_y,
                                                       self.selected_feature_th_dict[selected_feature])
    
            df_singlefeature_pred.append(y_pred + 0)
        df_singlefeature_pred = np.array(df_singlefeature_pred)
        y_pred = self.get_final_majority_voting(df_singlefeature_pred)
        return y_pred

    def predict_proba(self, X, y=None):
        df_singlefeature_pred = []
        assert set(self.selected_features).issubset(set(X.columns))
        for selected_feature in self.selected_features:
            wk_e, feat = selected_feature.split("::")
            df_userbehavior_memory = np.array(self.data_memory_X[selected_feature].to_list())
            df_userbehavior_target = np.array(X[selected_feature].to_list())
            sim_cor_square = (1 - pairwise_distances(X = df_userbehavior_target,
                                            Y = df_userbehavior_memory, metric = "correlation")).astype(np.float64)
            sim_cor_square = sim_cor_square * sim_cor_square

            y_pred = self.collaborative_filtering_pred(sim_cor_square, self.data_memory_y,
                                                       self.selected_feature_th_dict[selected_feature])
    
            df_singlefeature_pred.append(y_pred + 0)
        df_singlefeature_pred = np.array(df_singlefeature_pred)
        y_pred_prob = self.get_final_majority_voting_prob(df_singlefeature_pred)
        return y_pred_prob

class DepressionDetectionAlgorithm_ML_xu_personalized(DepressionDetectionAlgorithm_ML_basic):
    """Algirithm for Xu et al. personalized work, extending the basic traditional ml algorithm """

    def __init__(self, config_dict = None, config_name = "ml_xu_personalized"):
        super().__init__(config_dict, config_name)

        if (self.flag_use_norm_features):
            self.feature_list = [
                f"{f}_norm:{epoch}" for f in self.feature_list_base \
                    for epoch in epochs_5
            ]
            self.feature_list_epoch = { epoch: [
                f"{f}_norm:{epoch}" for f in self.feature_list_base \
                ] for epoch in epochs_5
            }
        else:
            self.feature_list = [
                f"{f}:{epoch}" for f in self.feature_list_base \
                    for epoch in epochs_5
            ]
            self.feature_list_epoch = { epoch: [
                f"{f}:{epoch}" for f in self.feature_list_base \
                ] for epoch in epochs_5
            }
        self.feature_dis_list_epoch = { epoch: [
            f"{f}_dis:{epoch}" for f in self.feature_list_base \
            ] for epoch in epochs_5
        }

    def unravel_df(self, row: pd.Series, flag_train: bool = False):
        X_raw = deepcopy(row["X_raw"])
        X_raw["pid"] = row["pid_new"]
        X_raw["pid_origin"] = row["pid"]
        X_raw["y_raw"] = row["y_raw"]
        return X_raw

    def data_conversion_imputation(self, df_dis: pd.DataFrame, flag_train:bool = False):
        """ Impute the data by collaborative filtering """
        def fill_by_CF(col_to_be_filled: list or np.ndarray, similarity_matrix: np.ndarray):
            column_value = np.array(col_to_be_filled)
            isna_tmp = pd.isna(column_value)
            idx_nan = np.argwhere(isna_tmp)[:,0]
            idx_nonnan = np.argwhere(~isna_tmp)[:,0]
            if (len(idx_nan) == 0):
                return pd.Series(column_value, index=col_to_be_filled.index)
            similarity_for_pids_nan = np_take(similarity_matrix, idx_nan, idx_nonnan)
            fill_scores = (similarity_for_pids_nan * column_value[idx_nonnan]).sum(axis = 1) / (similarity_for_pids_nan.sum(axis = 1) + 0.0001)
            fill_scores = np.clip(np.round(fill_scores),1,3)
            column_value[idx_nan] = fill_scores
            return pd.Series(column_value, index=col_to_be_filled.index)

        @globalize
        def filter_and_fill_col(df: pd.DataFrame, feat: str, flag_train: bool):
            signal.signal(signal.SIGTERM, lambda signalNumber, frame: False)
            df_pivot = pd.pivot_table(
                data = df,
                index = "pid", columns="#day", dropna = False)
            
            narate_date = df_pivot.isna().sum(axis = 0) / df_pivot.shape[0]
            narate_data_flag = narate_date < 0.75
            if flag_train:
                empty_feature_filtering_th = -1
            else:
                empty_feature_filtering_th = self.config["feature_definition"]["empty_feature_filtering_th"]
            if sum(narate_data_flag) > (len(narate_date) * empty_feature_filtering_th): # remove very empty features
                # avoid NA
                df_pivot_ = df_pivot.fillna(df_pivot.mean())
                df_pivot_ = df_pivot.fillna(0) # in case the data is NA
                df_pivot_.loc[(df_pivot_.apply(lambda x : len(set(x)), axis = 1) == 1),(feat,0)] += 0.0001
                similarity_with_nan = (1 - pairwise_distances(df_pivot_, metric = "correlation")).astype(np.float64)
                df_pivot = df_pivot.apply(fill_by_CF, axis = 0, similarity_matrix = similarity_with_nan)
                # avoid NA
                df_pivot.loc[(df_pivot.apply(lambda x : len(set(x)), axis = 1) == 1),(feat,0)] += 0.0001
                return df_pivot
            else:
                return None

        df_userbehavior_profile_perfeature = {w + "_" + e : {} for w in wks for e in epochs_5}
        for wk in wks:
            for e in epochs_5:
                wk_e = wk + "_" + e
                if (self.verbose > 0):
                    print("Imputation epoch: ", wk_e)
                
                # not using Ray as function memory could be huge
                with Pool(NJOB) as pool:
                    pool_results = pool.starmap(filter_and_fill_col,
                    [(df_dis[wk_e][["pid", "#day", f]], f, flag_train) for f in self.feature_dis_list_epoch[e]])

                for df_filled, f in zip(pool_results, self.feature_dis_list_epoch[e]):
                    if (df_filled is not None):
                        df_userbehavior_profile_perfeature[wk_e][f] = deepcopy(df_filled)
                    else:
                        if not flag_train:
                            df_tmp = pd.pivot_table(data = df_dis[wk_e][["pid", "#day", f]],
                                                index = "pid", columns="#day", dropna = False)
                            df_userbehavior_profile_perfeature[wk_e][f] = df_tmp.fillna(0)
        return df_userbehavior_profile_perfeature

    def prep_data_repo(self, dataset:DatasetDict, flag_train:bool = True) -> DataRepo:
        set_random_seed(42)

        Path(self.results_save_folder).mkdir(parents=True, exist_ok=True)
        self.save_file_path = os.path.join(self.results_save_folder, dataset.key + "--" + dataset.prediction_target + ".pkl")

        df_datapoint_extend = deepcopy(dataset.datapoints)
        df_datapoint_extend["pid_new"] = df_datapoint_extend["pid"] + "@" + df_datapoint_extend["date"].apply(lambda x : x.strftime("%Y-%m-%d"))
        df_datapoint_extend_unravel = df_datapoint_extend.apply(lambda row: self.unravel_df(row, flag_train), axis = 1)
        df_datapoint_extend = pd.concat(list(df_datapoint_extend_unravel))
        df_labels = df_datapoint_extend[["pid", "y_raw", "pid_origin"]].drop_duplicates(keep="first").set_index("pid")

        if (self.config["training_params"]["save_and_reload"] and os.path.exists(self.save_file_path)):
            with open(self.save_file_path, "rb") as f:
                data_repo_X = pickle.load(f)
        else:
            df_dis_epochs = {}
            for w in wks:
                for e in epochs_5:
                    wk_e = w + "_" + e
                    if (w == "wkdy"):
                        wk_idx = df_datapoint_extend["date"].dt.dayofweek < 5
                    else:
                        wk_idx = df_datapoint_extend["date"].dt.dayofweek >= 5
                    df_dis_epochs[wk_e] = deepcopy(df_datapoint_extend[wk_idx]\
                                                    [["pid", "date"] + self.feature_dis_list_epoch[e]])
                    df_dis_epochs[wk_e]["#day"] = df_dis_epochs[wk_e].groupby("pid").apply(lambda x : np.argsort(x["date"])).values
                    df_dis_epochs[wk_e][self.feature_dis_list_epoch[e]] = \
                        df_dis_epochs[wk_e][self.feature_dis_list_epoch[e]].apply(lambda col : col.map({"l":1, "m":2,"h":3}))

            df_userbehavior_profile_perfeature = self.data_conversion_imputation(df_dis_epochs, flag_train)

            # convert from feature-unit data to person-unit data
            data_repo_X = {}
            for wk_e, df_userbehavior_feats in df_userbehavior_profile_perfeature.items():
                for feat, df_userbehavior in df_userbehavior_feats.items():
                    key = f"{wk_e}::{feat}"
                    for pid, row in zip(df_userbehavior.index, df_userbehavior.values):
                        if pid not in data_repo_X:
                            data_repo_X[pid] = {}
                        data_repo_X[pid][key] = list(row)

            if (self.config["training_params"]["save_and_reload"]):
                with open(self.save_file_path, "wb") as f:
                    pickle.dump(data_repo_X, f)

        X = pd.DataFrame(data_repo_X).T

        y = df_labels["y_raw"][X.index]
        pids = df_labels["pid_origin"][X.index]
        self.prep_dataset_key = deepcopy(dataset.key)

        self.data_repo = DataRepo(X=X, y=y, pids=pids)
        self.data_repo.key = deepcopy(dataset.key)
        self.data_repo.prediction_target = deepcopy(dataset.prediction_target)
        return self.data_repo


    def get_singlefeature_results(self, data_repo: DataRepo):

        def collaborative_filtering_inner_cv(Y: list or np.ndarray, grps: list or np.ndarray, sim_mtx: np.ndarray):
            training_results = {}
            valid_pred_score_list = []
            valid_test_list = []
            valid_result = []
            train_score_list = []
            train_result = []
            train_labels_list = []
            ths_list = []
            th_list = []
            full_index = np.arange(len(Y))
            pidnum_min = get_min_count_class(labels = Y, groups = grps)
            
            for train_index, valid_index in GroupKFold(n_splits=min(40,pidnum_min)).split(full_index, groups = grps):
                Y_train, Y_valid = Y[train_index], Y[valid_index]
                sim_weight = np_take(sim_mtx,valid_index,train_index)
                sim_weight_abs = np.abs(sim_weight)
                zeroout_idx = np.where(sim_weight_abs < np.percentile(sim_weight_abs, 25))
                sim_weight[zeroout_idx] = 0
                valid_pred_score = np.matmul(sim_weight, Y_train)
                valid_pred_score /= (sim_weight.sum(axis=1) + 0.000001)
                valid_pred_score_list.append(valid_pred_score)
                valid_test_list.append(Y_valid)

                sim_mtx_buf = np_take(sim_mtx,train_index,train_index)
                np.fill_diagonal(sim_mtx_buf, 0)
                sim_mtx_buf_rowsum = sim_mtx_buf.sum(axis = 1)
                train_score = ((np.matmul(sim_mtx_buf, Y_train)) / (sim_mtx_buf_rowsum + 0.000001))
                
                train_score_list.append(deepcopy(train_score))
                train_labels_list.append(deepcopy(Y_train))

                train_score1 = train_score[np.where(Y_train)[0]]
                train_score2 = train_score[np.where(1 - Y_train)[0]]

                mean1 = np.mean(train_score1)
                mean2 = np.mean(train_score2)
                th = (mean1 + mean2) / 2
                
                y_pred_train = (train_score > th)
                
                d_train = utils_ml.results_report(y_test=Y_train, y_pred = y_pred_train, verbose=False)
                try:
                    rocauc = roc_auc_score(Y_train, train_score)
                except:
                    rocauc = None
                d_train.update({"rocauc":rocauc})
                train_result.append(d_train)
            
                y_pred_valid = (valid_pred_score > th)
                d_valid = utils_ml.results_report(y_test=Y_valid, y_pred = y_pred_valid, verbose=False)
                try:
                    rocauc = roc_auc_score(Y_valid, valid_pred_score)
                except:
                    rocauc = None
                d_valid.update({"rocauc":rocauc})
                valid_result.append(d_valid)

                ths_list.append([th] * len(Y_valid))
                th_list.append(th)
                
            training_results["th_list"] = np.array(th_list)
            
            train_result = pd.DataFrame(train_result)
            for col in train_result.columns:
                training_results["train_" + col] = train_result[col].to_list()
                
            valid_result = pd.DataFrame(valid_result)
            for col in valid_result.columns:
                training_results["valid_" + col] = valid_result[col].to_list()
            
            ths_list = np.concatenate(ths_list)
            valid_pred_score_list = np.concatenate(valid_pred_score_list)
            valid_test_list = np.concatenate(valid_test_list)
            
            y_pred_results = (valid_pred_score_list > ths_list)
            valid_results = utils_ml.results_report(y_test = valid_test_list, y_pred = y_pred_results, verbose=False)
            training_results["valid_results"] = valid_results

            return training_results

        @ray.remote(num_cpus=int(np.ceil(NJOB/2))) # save memory
        def get_singlefeature_results_func(data_repo, feat):
            df_userbehavior = np.array(data_repo.X[feat].to_list())
            y = data_repo.y
            grps = data_repo.pids

            sim_cor_square = (1 - pairwise_distances(df_userbehavior, metric = "correlation")).astype(np.float64)
            sim_cor_square = sim_cor_square * sim_cor_square
            
            training_results = collaborative_filtering_inner_cv(Y=y, grps=grps,
                                                    sim_mtx=sim_cor_square)
            return training_results

        df_singlefeature_results = {}

        data_repo_id = ray.put(data_repo)
        pool_results = ray.get([get_singlefeature_results_func.remote(data_repo_id, feat) for feat in data_repo.X.columns])

        for results, feat in zip(pool_results, data_repo.X.columns):
            df_singlefeature_results[feat] = deepcopy(results)
        return pd.DataFrame(df_singlefeature_results).T

    def select_features_from_allfeatures(self, df_singlefeature_results: pd.date_range, criteria: str = "balanced_acc"):
        def rank_features(df_singlefeature, select_ref_col):
            df_singlefeatures_filtered = df_singlefeature[df_singlefeature["train_rec"] < 0.99]
            selected_features = {}
            selected_acc_th = 0.53
            df_singlefeatures_filtered = df_singlefeatures_filtered.sort_values(select_ref_col, ascending = False)

            # Sort the global best features
            top_rows = df_singlefeatures_filtered[df_singlefeatures_filtered["valid_acc"] > selected_acc_th]
            return np.array([{"feat":i,"rank":j+1} for j,i in enumerate(list(top_rows.index))])


        features_ranks_cv = []
        for loop_idx in range(len(df_singlefeature_results["valid_acc"].iloc[0])):
            criteria_col = f"train_{criteria}"
            picked_cols = list(set(["valid_acc", "train_rec", criteria_col]))
            df_singlefeature_results_cv = deepcopy(df_singlefeature_results[picked_cols].apply(lambda row : pd.Series([r[loop_idx] for r in row])))
            df_singlefeature_results_cv.index = df_singlefeature_results.index
            features_ranks = rank_features(df_singlefeature_results_cv, f"train_{criteria}")
            features_ranks_cv.append(features_ranks)

        features_scores = {}
        for feature_ranks in features_ranks_cv:
            for feat_dict in feature_ranks:
                feat = feat_dict["feat"]
                rank = feat_dict["rank"]
                score = len(feature_ranks) - rank if (rank <= len(feature_ranks) // 5) else 0
                if (feat in features_scores):
                    features_scores[feat] += score
                else:
                    features_scores[feat] = score

        selected_features = {}
        seq_len = int(np.ceil(len(features_scores) * 0.05)) # top 5% features
        selected_features = \
            [k for k in sorted(features_scores, key=features_scores.get, reverse = True)][:seq_len]
        selected_features_th = {}
        for feat in selected_features:
            selected_features_th[feat] = df_singlefeature_results.loc[feat]["th_list"].mean()
        return selected_features_th


    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        super().prep_model()
        set_random_seed(42)
        assert hasattr(data_train, 'key')

        df_singlefeature_results_file_path = os.path.join(self.results_save_folder,
            data_train.key + "--" + data_train.prediction_target + "_df_singlefeature_results.pkl")
        if (self.config["training_params"]["save_and_reload"] and os.path.exists(df_singlefeature_results_file_path)):
            with open(df_singlefeature_results_file_path, "rb") as f:
                df_singlefeature_results = pickle.load(f)
        else:
            df_singlefeature_results = self.get_singlefeature_results(data_train)
            if (self.config["training_params"]["save_and_reload"]):
                with open(df_singlefeature_results_file_path, "wb") as f:
                    pickle.dump(df_singlefeature_results, f)

        selected_features = self.select_features_from_allfeatures(df_singlefeature_results, criteria = criteria)

        clf = DepressionDetectionClassifier_ML_xu_personalized(selected_features)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        r = cross_validate(clf, X=data_train.X[selected_features], y=data_train.y, groups= data_train.pids, cv = cv,
                scoring = utils_ml.results_report_sklearn, return_train_score=False)
        r = {k:np.mean(v) for k,v in r.items()}
        if (self.verbose > 0):
            print(r)

        return DepressionDetectionClassifier_ML_xu_personalized(selected_features)
