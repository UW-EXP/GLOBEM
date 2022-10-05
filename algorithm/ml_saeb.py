"""
Implementation of prior depression detection algorithm:

Sohrab Saeb, Mi Zhang, Christopher J. Karr, Stephen M. Schueller,
Marya E. Corden, Konrad P. Kording, and David C. Mohr. 2015.
Mobile phone sensor correlates of depressive symptom severity in daily-life behavior: An exploratory study.
Journal of Medical Internet Research 17, 7 (2015), 1–11. https://doi.org/10.2196/jmir.4273
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.ml_basic import DepressionDetectionAlgorithm_ML_basic
from data_loader.data_loader_ml import DataRepo
from algorithm.base import DepressionDetectionClassifierBase

class DepressionDetectionClassifier_ML_saeb(DepressionDetectionClassifierBase):
    """Classifier for Saeb et al. work. Train a simple logistic regression model with elastic net regularization. """
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

class DepressionDetectionAlgorithm_ML_saeb(DepressionDetectionAlgorithm_ML_basic):
    """Algirithm for Saeb et al. work, extending the basic traditional ml algorithm """

    def __init__(self, config_dict = None, config_name = "ml_saeb"):
        super().__init__(config_dict, config_name)

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
                X_tmp = df_datapoints["X_raw"].swifter.progress_bar(False).apply(lambda x :
                    pd.Series(data = x[self.feature_list].iloc[-14:].mean().values, index = self.feature_list).T)

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

            clf = utils_ml.get_clf("lr", model_parameters, direct_param_flag = True)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            r = cross_validate(clf, X=X, y=y, groups= pids, cv = cv,
                    scoring = utils_ml.results_report_sklearn, return_train_score=False)
            r = {k:np.mean(v) for k,v in r.items()}
            r.update({"parameters":model_parameters})
            return r

        l1_ratio_list = [0,0.5,1]
        C_list = [0.1,1,10]
        parameters_list = []
        for l1_ratio, C in itertools.product(l1_ratio_list, C_list):
            parameters_tmp = {"penalty":"elasticnet","max_iter":10000, "solver":"saga","l1_ratio":l1_ratio,"C":C, "random_state":42}
            parameters_list.append(parameters_tmp)

        data_train_id = ray.put(data_train)
        results_list = ray.get([train_small_cv.remote(data_train_id,i) for i in parameters_list])
        results_list = pd.DataFrame(results_list)

        best_row = results_list.iloc[results_list[f"test_{criteria}"].argmax()]
        best_params = best_row['parameters']
        if (self.verbose > 0):
            print(best_row)
            print(best_params)
        
        return DepressionDetectionClassifier_ML_saeb(model_params=best_params, selected_features=data_train.X.columns)
