"""
Implementation of prior depression detection algorithm:

Jin Lu, Jinbo Bi, Chao Shang, Chaoqun Yue, Reynaldo Morillo, Shweta Ware,
Jayesh Kamath, Athanasios Bamis, Alexander Russell, and Bing Wang. 2018.
Joint Modeling of Heterogeneous Sensing Data for Depression Assessment via Multi-task Learning.
Proceedings of the ACM on Interactive, Mobile,Wearable and Ubiquitous Technologies 2, 1 (2018) 1–21
https://doi.org/10.1145/3191753 ISBN: 9781450351980.
"""


import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.ml_basic import DepressionDetectionAlgorithm_ML_basic
from data_loader.data_loader_ml import DataRepo
from algorithm.base import DepressionDetectionClassifierBase
from utils.cv_split import judge_corner_cvsplit

class DepressionDetectionClassifier_ML_lu(DepressionDetectionClassifierBase):
    """Classifier for Lu et al. work. Fits two models based on device type with a MTL framework"""
    def __init__(self, gamma1, gamma2, selected_features, p = 1, k = 2):
        self.device_type_set = set([0,1]) # ios - 1, android - 0
        self.selected_features = selected_features
        self.num_tasks = 2
        self.tolerance = 0.01
        self.max_iter = 10
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.p = p
        self.k = k
        assert self.p in [1,2]
        assert self.k in [1,2]
        
        self.c_history = []
        self.alpha_history = []
        self.beta_history = []

        self.no_data_flag_ios = False
        self.no_data_flag_android = False
    
    def fit(self, X, y=None):
        assert set(self.selected_features).issubset(set(X.columns))
        assert X.columns[-1] == "device_type"
        set_random_seed(42)
        X_np = np.array(X)
        y_np = np.array(y)
        devices = X_np[:,-1]
        assert set(devices).issubset(self.device_type_set)
        self.feature_num = X_np.shape[-1] - 1
        devices_ios_index = np.where(devices == 1)[0]
        devices_android_index = np.where(devices != 1)[0]
        # Task 1
        X_ios = X_np[devices_ios_index,:-1]
        y_ios = y_np[devices_ios_index]
        # Task 2
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

        X_tasks = []
        y_tasks = []
        if (not self.no_data_flag_ios):
            X_tasks.append(X_ios)
            y_tasks.append(y_ios)
        if (not self.no_data_flag_android):
            X_tasks.append(X_android)
            y_tasks.append(y_android)
        
        self.c_diag = np.ones(self.feature_num)
        self.c = np.diag(self.c_diag)
        self.c_history.append(deepcopy(self.c_diag))
        
        self.blockwise_coordinate_descent(X_tasks, y_tasks)
        return self.find_best_threshold(X_tasks, y_tasks)
        
    def blockwise_coordinate_descent(self, X_tasks, y_tasks):
        while True:
            beta_tasks = []
            alpha_tasks = []
            for task in range(len(X_tasks)):
                X_t = X_tasks[task]
                y_t = y_tasks[task]
                X_t_hat = np.matmul(X_t, self.c)
                clf = linear_model.LogisticRegression(penalty = "l1" if self.p == 1 else "l2",
                                                      C = self.gamma1, solver = "saga", random_state = 42,
                                                      max_iter=30000, tol = 0.001)
                clf.fit(X_t_hat, y_t)
                beta_t = deepcopy(clf.coef_).T
                alpha_t = np.matmul(self.c, beta_t)
                beta_tasks.append(beta_t)
                alpha_tasks.append(alpha_t)
            alpha_tasks = np.array(alpha_tasks)
            self.beta = np.matrix(np.concatenate(beta_tasks, axis = 1))
            self.alpha = np.matrix(np.concatenate(alpha_tasks, axis = 1))
            self.beta_history.append(self.beta)
            self.alpha_history.append(self.alpha)
            c_diag_new = np.empty(len(self.c))
            s = (alpha_tasks ** self.p).sum(axis = 0)
            c_diag_new = (self.gamma1 * np.abs(s) / self.gamma2) ** (1.0 / (self.p + self.k)) * np.sign(s)
            self.c_diag = c_diag_new[:,0]
            self.c = np.diag(self.c_diag)
            self.c_history.append(deepcopy(self.c_diag))
            if (len(self.alpha_history) > 2):
                if (len(self.alpha_history) > self.max_iter):
                    # print("Exceed max iter!", self.gamma1, self.gamma2)
                    break
                delta = np.linalg.norm(self.alpha - self.alpha_history[-2], ord = 1)
                if (delta < self.tolerance):
                    # print("iter num", len(self.alpha_history))
                    break
    
    def find_best_threshold(self, X_tasks, y_tasks):
        y_prob_train = []
        y_test = []
        best_th_tasks = []
        for task in range(len(X_tasks)):
            X_t = X_tasks[task]
            y_t = y_tasks[task]
            alpha = self.alpha_history[-1][:,task]
            y_prob = 1 / (1 + np.exp(-np.asarray(np.matmul(X_t, alpha))[:,0]))
            y_prob_train.append(y_prob)
            y_test.append(y_t)
            roc = sklearn.metrics.roc_curve(y_true = y_t, y_score = y_prob)
            best_th = roc[2][np.argmax((1 - roc[0] + roc[1]) / 2)] # select based on bal acc
            best_th_tasks.append(best_th)
        
        y_prob_train = np.concatenate(y_prob_train)
        y_test = np.concatenate(y_test)
        roc = sklearn.metrics.roc_curve(y_true = y_test, y_score = y_prob_train)
        best_th = roc[2][np.argmax((1 - roc[0] + roc[1]) / 2)] # select based on bal acc
        self.best_th_single = best_th
        self.best_th_tasks = deepcopy(best_th_tasks)
        
    
    def predict_complete(self, X, y=None):
        X_np = np.array(X)
        y_pred = np.empty(len(X_np))
        y_pred_prob = np.empty((len(X_np),2))
        devices = X_np[:,-1]
        devices_ios_index = np.where(devices == 1)[0]
        devices_android_index = np.where(devices != 1)[0]
        
        X_np = np.array(X)
        y_np = np.array(y)
        devices = X_np[:,-1]
        devices_ios_index = np.where(devices == 1)[0]
        devices_android_index = np.where(devices != 1)[0]
        X_ios = X_np[devices_ios_index,:-1]
        X_android = X_np[devices_android_index,:-1]

        X_tasks = []
        idx_tasks = []
        if (len(devices_ios_index) > 0):
            X_tasks.append(X_ios)
            idx_tasks.append(devices_ios_index)
        if (len(devices_android_index) > 0):
            X_tasks.append(X_android)
            idx_tasks.append(devices_android_index)
        
        for task, (X_t, idx_t) in enumerate(zip(X_tasks, idx_tasks)):
            alpha = self.alpha_history[-1][:,task]

            y_prob = 1 / (1 + np.exp(-np.asarray(np.matmul(X_t, alpha))[:,0]))
            best_th = self.best_th_tasks[task]
            y_pred[idx_t] = y_prob > best_th

            y_prob_tmp = 1 / (1 + np.exp(-np.asarray(np.matmul(X_t, alpha))[:,0]))
            y_prob_list = np.concatenate([[1 - y_prob_tmp], [y_prob_tmp]]).T
            y_pred_prob[idx_t] = y_prob_list

        return y_pred, y_pred_prob


    def predict(self, X, y=None):
        y_pred, y_pred_prob = self.predict_complete(X, y)
        return y_pred

    def predict_proba(self, X, y=None):
        y_pred, y_pred_prob = self.predict_complete(X, y)
        return y_pred_prob

class DepressionDetectionAlgorithm_ML_lu(DepressionDetectionAlgorithm_ML_basic):
    """Algirithm for Lu et al. work, extending the basic traditional ml algorithm """

    def __init__(self, config_dict = None, config_name = "ml_lu"):
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

            model_parameters_full = deepcopy(model_parameters)
            model_parameters_full.update({"selected_features": X.columns})
            clf = DepressionDetectionClassifier_ML_lu(**model_parameters_full)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            repeat_time = 0
            while True:
                repeat_time += 1
                cv = StratifiedGroupKFold(n_splits=5,shuffle=True, random_state=42+repeat_time*1000)
                if (judge_corner_cvsplit(cv, data_repo)):
                    continue
                else:
                    break
            r = cross_validate(clf, X=X, y=y, groups=pids, cv = cv,
                    scoring = utils_ml.results_report_sklearn,
                    return_train_score=False)
            r = {k:np.mean(v) for k,v in r.items()}
            r.update({"parameters":model_parameters})
            return r

        gamma1_list = [10**i for i in range(-2,3)]
        gamma2_list = [10**i for i in range(-2,3)]
        p_list = [1] # [1,2]
        k_list = [1] # [2,1]
        parameters_list = []
        for gamma1, gamma2, p, k in itertools.product(gamma1_list, gamma2_list, p_list, k_list):
            parameters_tmp = {"gamma1": gamma1, "gamma2": gamma2, "p":p, "k":k}
            parameters_list.append(parameters_tmp)

        data_train_id = ray.put(data_train)
        results_list = ray.get([train_small_cv.remote(data_train_id,i) for i in parameters_list])
        results_list = pd.DataFrame(results_list)

        best_row = results_list.iloc[results_list[f"test_{criteria}"].argmax()]
        best_params = best_row['parameters']
        if (self.verbose > 0):
            print(best_row)
            print(best_params)
        
        best_params_full = deepcopy(best_params)
        best_params_full.update({"selected_features": data_train.X.columns})
        return DepressionDetectionClassifier_ML_lu(**best_params_full)