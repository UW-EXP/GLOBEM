"""
Implementation of prior depression detection algorithm:

Luca Canzian and Mirco Musolesi. 2015. 
Trajectories of depression: Unobtrusive monitoring of depressive states by means of smartphone mobility traces analysis. 
Proceedings of the ACM International Joint Conference on Pervasive and Ubiquitous Computing (2015), 1293–1304. 
https://doi.org/10.1145/2750858.2805845
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *

from algorithm.base import DepressionDetectionClassifierBase
from algorithm.ml_basic import DepressionDetectionAlgorithm_ML_basic
from data_loader.data_loader_ml import DataRepo

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

class DepressionDetectionClassifier_ML_canzian(DepressionDetectionClassifierBase):
    """Classifier for Canzian et al. work. A simple SVM classifier """
    def __init__(self, model_params, selected_features):
        self.model_params = model_params
        self.selected_features = selected_features
        
        self.clf = utils_ml.get_clf("svm", model_params, direct_param_flag = True)
    def fit(self, X, y):
        set_random_seed(42)
        assert set(self.selected_features).issubset(set(X.columns))
        return self.clf.fit(X[self.selected_features], y)
    def predict(self, X, y=None):
        return self.clf.predict(X[self.selected_features])
    def predict_proba(self, X, y=None):
        return self.clf.predict_proba(X[self.selected_features])

class DepressionDetectionAlgorithm_ML_canzian(DepressionDetectionAlgorithm_ML_basic):
    """Algirithm for Canzian et al. work, extending the basic traditional ml algorithm """

    def __init__(self, config_dict = None, config_name = "ml_canzian"):
        super().__init__(config_dict, config_name)
    
    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        super().prep_model()
        set_random_seed(42)

        @ray.remote
        def train_small_cv(data_repo: DataRepo, model_parameters: dict):
            warnings.filterwarnings("ignore")
            X = data_repo.X
            y = data_repo.y
            pids = data_repo.pids

            clf = utils_ml.get_clf("svm", model_parameters, direct_param_flag = True)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            r = cross_validate(clf, X=X, y=y, groups= pids, cv = cv,
                    scoring = utils_ml.results_report_sklearn_noprob, return_train_score=False)
            r = {k:np.mean(v) for k,v in r.items()}
            r.update({"parameters":model_parameters})
            return r

        C_list = [0.03125,0.125,0.5,2,8,32]
        parameters_list = []
        for C in C_list:
            parameters_tmp = {"kernel":"rbf", "max_iter":50000,"C":C,"random_state":42}
            parameters_list.append(parameters_tmp)

        data_train_id = ray.put(data_train)
        results_list = ray.get([train_small_cv.remote(data_train_id,i) for i in parameters_list])
        results_list = pd.DataFrame(results_list)

        best_row = results_list.iloc[results_list[f"test_{criteria}"].argmax()]
        best_params = best_row['parameters']
        if (self.verbose > 0):
            print(best_row)
            print(best_params)
        best_params["probability"] = True
        return DepressionDetectionClassifier_ML_canzian(model_params=best_params, selected_features=data_train.X.columns)
