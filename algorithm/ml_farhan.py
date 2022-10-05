"""
Implementation of prior depression detection algorithm:

Asma Ahmad Farhan, Chaoqun Yue, Reynaldo Morillo, Shweta Ware, Jin Lu, Jinbo Bi,
Jayesh Kamath, Alexander Russell, Athanasios Bamis, and Bing Wang. 2016. 
Behavior vs. introspection: refining prediction of clinical depression via smartphone sensing data.
In 2016 IEEE Wireless Health (WH). IEEE, 1–8. https://doi.org/10.1109/WH.2016.7764553
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.ml_basic import DepressionDetectionAlgorithm_ML_basic
from data_loader.data_loader_ml import DataRepo
from algorithm.base import DepressionDetectionClassifierBase

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

class DepressionDetectionClassifier_ML_farhan(DepressionDetectionClassifierBase):
    """Classifier for Farhan et al. work. Two SVM, one for ios, and the other for android """
    def __init__(self, model_params, selected_features):
        self.model_params = model_params
        self.selected_features = selected_features
        self.svm_ios = utils_ml.get_clf("svm", model_params, direct_param_flag = True)
        self.svm_android = utils_ml.get_clf("svm", model_params, direct_param_flag = True)
        self.device_type_set = set([0,1]) # ios - 1, android - 0
        self.single_class_flag_android = False
        self.single_class_android = None
        self.single_class_flag_ios = False
        self.single_class_ios = None
        self.no_data_flag_ios = False
        self.no_data_flag_android = False
    
    
    def fit(self, X, y=None):
        assert set(self.selected_features).issubset(set(X.columns))
        assert X.columns[-1] == "device_type"
        set_random_seed(42)
        X_np = np.array(X[self.selected_features])
        y_np = np.array(y)
        devices = X_np[:,-1]
        assert set(devices).issubset(self.device_type_set)
        devices_ios_index = np.where(devices == 1)[0]
        devices_android_index = np.where(devices != 1)[0]
        X_ios = X_np[devices_ios_index,:-1]
        y_ios = y_np[devices_ios_index]

        X_android = X_np[devices_android_index,:-1]
        y_android = y_np[devices_android_index]

        if (len(devices_ios_index) == 0):
            self.no_data_flag_ios = True
        else:
            self.no_data_flag_ios = False
        if (len(devices_android_index) == 0):
            self.no_data_flag_android = True
        else:
            self.no_data_flag_android = False
        
        if (not self.no_data_flag_ios):
            if (len(np.unique(y_ios)) == 1):
                self.single_class_flag_ios = True
                self.single_class_ios = y_ios[0]
            else:
                self.svm_ios.fit(X_ios, y_ios)
            if (self.no_data_flag_android): # only ios data
                self.svm_android = self.svm_ios

        if (not self.no_data_flag_android):
            if (len(np.unique(y_android)) == 1):
                self.single_class_flag_android = True
                self.single_class_android = y_android[0]
            else:
                self.svm_android.fit(X_android, y_android)
            if (self.no_data_flag_ios): # only android data
                self.svm_android = self.svm_android
    
    def predict(self, X, y=None):
        X_np = np.array(X)
        y_pred = np.empty(len(X_np))
        devices = X_np[:,-1]
        devices_ios_index = np.where(devices == 1)[0]
        devices_android_index = np.where(devices != 1)[0]
        
        if (len(devices_ios_index) > 0):
            X_ios = X_np[devices_ios_index,:-1]
            if (self.single_class_flag_ios):
                y_pred[devices_ios_index] = np.array([self.single_class_ios for _ in range(len(X_ios))])
            else:
                y_pred[devices_ios_index] = self.svm_ios.predict(X_ios)
            
        if (len(devices_android_index) > 0):
            X_android = X_np[devices_android_index,:-1]
            if (self.single_class_flag_android):
                y_pred[devices_android_index] = np.array([self.single_class_android for _ in range(len(X_android))])
            else:
                y_pred[devices_android_index] = self.svm_android.predict(X_android)
        return y_pred
    
    def predict_proba(self, X, y=None):
        X_np = np.array(X)
        y_pred_prob = np.empty((len(X_np),2))
        devices = X_np[:,-1]
        devices_ios_index = np.where(devices == 1)[0]
        devices_android_index = np.where(devices != 1)[0]
        
        if (len(devices_ios_index) > 0):
            X_ios = X_np[devices_ios_index,:-1]
            if (self.single_class_flag_ios):
                y_pred_prob[devices_ios_index] = np.array([[0,1] if self.single_class_ios else [1,0] for _ in range(len(X_ios))])
            else:
                y_pred_prob[devices_ios_index] = self.svm_ios.predict_proba(X_ios)
            
        if (len(devices_android_index) > 0):
            X_android = X_np[devices_android_index,:-1]
            if (self.single_class_flag_android):
                y_pred_prob[devices_android_index] = np.array([[0,1] if self.single_class_android else [1,0] for _ in range(len(X_android))])
            else:
                y_pred_prob[devices_android_index] = self.svm_android.predict_proba(X_android)
        return y_pred_prob

class DepressionDetectionAlgorithm_ML_farhan(DepressionDetectionAlgorithm_ML_basic):
    """Algirithm for Farhan et al. work, extending the basic traditional ml algorithm """

    def __init__(self, config_dict = None, config_name = "ml_farhan"):
        super().__init__(config_dict, config_name)
    
    def prep_model(self, data_train: DataRepo, criteria: str = "f1") -> sklearn.base.ClassifierMixin:
        super().prep_model()
        set_random_seed(42)

        @ray.remote
        def train_small_cv(data_repo: DataRepo, model_parameters: dict):
            warnings.filterwarnings("ignore")
            X = data_repo.X
            y = data_repo.y
            pids = data_repo.pids

            clf = DepressionDetectionClassifier_ML_farhan(model_params=model_parameters, selected_features=X.columns)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            r = cross_validate(clf, X=X, y=y, groups=pids, cv = cv,
                    scoring = utils_ml.results_report_sklearn_noprob, return_train_score=False)
            r = {k:np.mean(v) for k,v in r.items()}
            r.update({"parameters":model_parameters})
            return r

        C_list = [2**i for i in range(-15,16)]
        # gamma_list = [2**i for i in range(-15,16)]
        gamma_list = ["scale"]
        parameters_list = []
        for C, gamma in itertools.product(C_list, gamma_list):
            parameters_tmp = {"kernel":"rbf", "max_iter":100000,"C":C, "gamma": gamma, "cache_size":700, "random_state":42}
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
        return DepressionDetectionClassifier_ML_farhan(model_params=best_params, selected_features=data_train.X.columns)
