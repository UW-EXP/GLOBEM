"""
Implementation of prior depression detection algorithm:

Fabian Wahle, Tobias Kowatsch, Elgar Fleisch, Michael Rufer, and Steffi Weidt. 2016.
Mobile Sensing and Support for People With Depression: A Pilot Trial in the Wild.
JMIR mHealth and uHealth 4, 3 (2016), e111. https://doi.org/10.2196/mhealth.5960 ISBN: doi:10.2196/mhealth.5960.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.ml_basic import DepressionDetectionAlgorithm_ML_basic
from data_loader.data_loader_ml import DataRepo
from algorithm.base import DepressionDetectionClassifierBase

class DepressionDetectionClassifier_ML_wahle(DepressionDetectionClassifierBase):
    """Classifier for Saeb et al. work. Train a SVM or a random forest model, controlled by model_type. """
    def __init__(self, model_params, model_type, selected_features):
        self.model_params = model_params
        self.model_type = model_type
        self.selected_features = selected_features
        self.clf = utils_ml.get_clf(self.model_type, self.model_params, direct_param_flag = True)
    def fit(self, X, y):
        assert set(self.selected_features).issubset(set(X.columns))
        set_random_seed(42)
        return self.clf.fit(X[self.selected_features], y)
    def predict(self, X, y=None):
        return self.clf.predict(X[self.selected_features])
    def predict_proba(self, X, y=None):
        return self.clf.predict_proba(X[self.selected_features])

class DepressionDetectionAlgorithm_ML_wahle(DepressionDetectionAlgorithm_ML_basic):
    """Algirithm for Wahle et al. work, extending the basic traditional ml algorithm """

    def __init__(self, config_dict = None, config_name = "ml_wahle"):
        super().__init__(config_dict, config_name)

        if (self.flag_use_norm_features):
            self.feature_list_direct = [col + "_norm:14dhist" for col in self.feature_list_base]
            self.feature_list_stats = [col + "_norm:allday" for col in self.feature_list_base]
        else:
            self.feature_list_direct = [col + ":14dhist" for col in self.feature_list_base]
            self.feature_list_stats = [col + ":allday" for col in self.feature_list_base]
            

    def prep_X(self, key:str, pred_target: str, df_datapoints: pd.DataFrame):
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
                X_tmp_direct = df_datapoints["X_raw"].apply(lambda x :
                    pd.Series(data = x[self.feature_list_direct].iloc[-1].values, index = self.feature_list_direct).T)
                X_tmp_mean = df_datapoints["X_raw"].swifter.progress_bar(False).apply(lambda x :
                    pd.Series(data = x[self.feature_list_stats].iloc[-14:].mean().values, index = [f + "#mean" for f in self.feature_list_stats]).T)
                X_tmp_sum = df_datapoints["X_raw"].swifter.progress_bar(False).apply(lambda x :
                    pd.Series(data = x[self.feature_list_stats].iloc[-14:].sum().values, index = [f + "#sum" for f in self.feature_list_stats]).T)
                X_tmp_std = df_datapoints["X_raw"].swifter.progress_bar(False).apply(lambda x :
                    pd.Series(data = x[self.feature_list_stats].iloc[-14:].std().values, index = [f + "#std" for f in self.feature_list_stats]).T)
                X_tmp_min = df_datapoints["X_raw"].swifter.progress_bar(False).apply(lambda x :
                    pd.Series(data = x[self.feature_list_stats].iloc[-14:].min().values, index = [f + "#min" for f in self.feature_list_stats]).T)
                X_tmp_max = df_datapoints["X_raw"].swifter.progress_bar(False).apply(lambda x :
                    pd.Series(data = x[self.feature_list_stats].iloc[-14:].max().values, index = [f + "#max" for f in self.feature_list_stats]).T)
                
                X_tmp = pd.concat([X_tmp_direct, X_tmp_mean, X_tmp_sum, X_tmp_std, X_tmp_min, X_tmp_max], axis = 1)

                if (self.config["training_params"]["save_and_reload"]):
                    with open(self.save_file_path, "wb") as f:
                        pickle.dump(X_tmp, f)

        return X_tmp
    
    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        super().prep_model()
        set_random_seed(42)

        @ray.remote
        def train_small_cv(data_repo: DataRepo, model_parameters: dict):
            warnings.filterwarnings("ignore")
            X = data_repo.X
            y = data_repo.y
            pids = data_repo.pids

            model_parameters_ = deepcopy(model_parameters)
            model_type = model_parameters_.pop("model_type")

            clf = utils_ml.get_clf(model_type, model_parameters_, direct_param_flag = True)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            r = cross_validate(clf, X=X, y=y, groups= pids, cv = cv,
                    scoring = utils_ml.results_report_sklearn_noprob, return_train_score=False)
            r = {k:np.mean(v) for k,v in r.items()}
            r.update({"parameters":model_parameters})
            return r

        parameters_list = []
        C_list = [0.015625, 0.03125,0.125,0.5,2,8,32,64]
        for C in C_list:
            parameters_tmp = {"model_type":"svm", "kernel":"rbf", "max_iter":50000,"C":C, "random_state":42}
            parameters_list.append(parameters_tmp)
        max_leaf_nodes_list = [2,4,6,8,10,12,16,32]
        for max_leaf_nodes in max_leaf_nodes_list:
            parameters_tmp = {"model_type":"rf", "n_estimators":450, "max_leaf_nodes":max_leaf_nodes, "random_state":42}
            parameters_list.append(parameters_tmp)

        data_train_id = ray.put(data_train)
        results_list = ray.get([train_small_cv.remote(data_train_id,i) for i in parameters_list])
        results_list = pd.DataFrame(results_list)

        best_row = results_list.iloc[results_list[f"test_{criteria}"].argmax()]
        best_params = best_row['parameters']
        if (self.verbose > 0):
            print(best_row)
            print(best_params)
        model_type = best_params.pop("model_type")
        if (model_type == "svm"):
            best_params["probability"] = True
        return DepressionDetectionClassifier_ML_wahle(model_params=best_params, model_type=model_type, selected_features=data_train.X.columns)
