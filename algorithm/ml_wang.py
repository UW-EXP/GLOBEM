"""
Implementation of prior depression detection algorithm:

Rui Wang, Weichen Wang, Alex daSilva, Jeremy F. Huckins, William M. Kelley,
Todd F. Heatherton, and Andrew T. Campbell. 2018.
Tracking Depression Dynamics in College Students Using Mobile Phone and Wearable Sensing.
Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 2, 1 (2018), 1–26.
https://doi.org/10.1145/3191775 ISBN: 2474-9567.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.ml_basic import DepressionDetectionAlgorithm_ML_basic
from data_loader.data_loader_ml import DataRepo
from algorithm.base import DepressionDetectionClassifierBase

class DepressionDetectionClassifier_ML_wang(DepressionDetectionClassifierBase):
    """Classifier for Saeb et al. work. Train a logistic regression model with L1 regularization. """
    def __init__(self, model_params, selected_features):
        self.model_params = model_params
        self.selected_features = selected_features
        
        self.clf = utils_ml.get_clf("lr", model_params, direct_param_flag = True)
    def fit(self, X, y):
        assert set(self.selected_features).issubset(set(X.columns))
        set_random_seed(42)
        return self.clf.fit(X[self.selected_features], y)
    def predict(self, X, y=None):
        return self.clf.predict(X[self.selected_features])
    def predict_proba(self, X, y=None):
        return self.clf.predict_proba(X[self.selected_features])


class DepressionDetectionAlgorithm_ML_wang(DepressionDetectionAlgorithm_ML_basic):
    """Algirithm for Wang et al. work, extending the basic traditional ml algorithm """

    def __init__(self, config_dict = None, config_name = "ml_wang"):
        super().__init__(config_dict, config_name)

    def get_slope(self, df_tmp: pd.DataFrame):
        df_tmp["lm_x"] = (df_tmp["date"] - df_tmp["date"].iloc[0]).apply(lambda x : x.days)

        slope_list = []
        for f in self.feature_list:
            idx_nonna = (~df_tmp[["lm_x", f]].isna()).all(axis = 1)
            if (sum(idx_nonna) == 0):
                slope_list.append(np.nan)
            else:
                # Alternative: use Sklearn LinearRegression
                # slope_list.append(linear_model.LinearRegression().fit(df_tmp[["lm_x"]][idx_nonna],df_tmp[f][idx_nonna]).coef_[0])
                x = df_tmp[["lm_x"]][idx_nonna].values
                y = df_tmp[f][idx_nonna].values
                sum_x = sum(x)
                sum_y = sum(y)
                beta = float((len(x) * np.matmul(x.T,y) - sum_x * sum_y) / (len(x) * np.matmul(x.T,x) - sum_x * sum_x + 1e-7)[0][0])
                slope_list.append(beta)
        return pd.Series(slope_list, index = [f + "#slope" for f in self.feature_list])

    def prep_X(self, key:str, pred_target:str, df_datapoints: pd.DataFrame):
        Path(self.results_save_folder).mkdir(parents=True, exist_ok=True)
        self.save_file_path = os.path.join(self.results_save_folder, key + "--" + pred_target + ".pkl")

        if (self.config["training_params"]["save_and_reload"] and os.path.exists(self.save_file_path)):
            with open(self.save_file_path, "rb") as f:
                X_tmp = pickle.load(f)
        else:
            flag_seperate_reload, X_tmp_reload = self.prep_X_reload(key, pred_target)
            if flag_seperate_reload:
                X_tmp = X_tmp_reload
            else:
                X_tmp_mean = df_datapoints["X_raw"].swifter.progress_bar(False).apply(lambda x :
                    pd.Series(data = x[self.feature_list].iloc[-14:].mean().values, index = [f + "#mean" for f in self.feature_list]).T)
                X_tmp_std = df_datapoints["X_raw"].swifter.progress_bar(False).apply(lambda x :
                    pd.Series(data = x[self.feature_list].iloc[-14:].std().values, index = [f + "#std" for f in self.feature_list]).T)

                @ray.remote
                def get_slope_ray(df):
                    return self.get_slope(deepcopy(df.iloc[-14:]))
                df_datapoints_Xraw_id_list = [ray.put(i) for i in df_datapoints["X_raw"]]
                X_tmp_slope = ray.get([get_slope_ray.remote(df) for df in df_datapoints_Xraw_id_list])
                X_tmp_slope = pd.DataFrame(X_tmp_slope)
                X_tmp_slope.index = X_tmp_mean.index
                for _ in range(len(df_datapoints)):
                    del df_datapoints_Xraw_id_list[-1]
                del df_datapoints_Xraw_id_list
                gc.collect()
                
                X_tmp = pd.concat([X_tmp_mean, X_tmp_std, X_tmp_slope], axis = 1)

                if (self.config["training_params"]["save_and_reload"]):
                    with open(self.save_file_path, "wb") as f:
                        pickle.dump(X_tmp, f)

        return X_tmp

    def prep_model(self, data_train: DataRepo, criteria: str = "deviance") -> sklearn.base.ClassifierMixin:
        super().prep_model()
        set_random_seed(42)
        
        @ray.remote
        def train_small_cv(data_repo: DataRepo, model_parameters: dict):
            warnings.filterwarnings("ignore")
            def results_report_sklearn_wang(clf, X, y):
                r_tmp = utils_ml.results_report_sklearn(clf, X, y)
                y_pred_p = clf.predict_proba(X)[:,1]
                r_tmp.update({"deviance": (y * np.log(y_pred_p) + (1-y)*np.log(1-y_pred_p)).sum()})
                return r_tmp

            X = data_repo.X
            y = data_repo.y
            pids = data_repo.pids

            clf = utils_ml.get_clf("lr", model_parameters, direct_param_flag = True)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            r = cross_validate(clf, X=X, y=y, groups= pids, cv = cv,
                    scoring = results_report_sklearn_wang, return_train_score=False)
            r = {k:np.mean(v) for k,v in r.items()}
            r.update({"parameters":model_parameters})
            return r

        C_list = [2**i for i in range(-15,16)]
        solver_list = ["saga"]
        parameters_list = []
        for C, solver in itertools.product(C_list, solver_list):
            parameters_tmp = {"penalty":"l1","max_iter":50000, "solver":solver,"C":C, "random_state":42}
            parameters_list.append(parameters_tmp)

        data_train_id = ray.put(data_train)
        results_list = ray.get([train_small_cv.remote(data_train_id,i) for i in parameters_list])
        results_list = pd.DataFrame(results_list)

        best_row = results_list.iloc[results_list[f"test_{criteria}"].argmax()]
        best_params = best_row['parameters']
        if (self.verbose > 0):
            print(best_row)
            print(best_params)
        
        return DepressionDetectionClassifier_ML_wang(model_params=best_params, selected_features=data_train.X.columns)
