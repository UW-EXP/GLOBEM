"""
Implementation of prior depression detection algorithm:

Prerna Chikersal, Afsaneh Doryab, Michael Tumminia, Daniella K Villalba, Janine M Dutcher, Xinwen Liu, Sheldon Cohen, Kasey G.
Creswell, Jennifer Mankoff, J. David Creswell, Mayank Goel, and Anind K. Dey. 2021.
Detecting Depression and Predicting its Onset Using Longitudinal Symptoms Captured by Passive Sensing.
ACM Transactions on Computer-Human Interaction 28, 1 (Jan. 2021), 1–41. https://doi.org/10.1145/3422821
"""

import os, sys
import six, numbers
from abc import ABCMeta, abstractmethod
from joblib import Memory, Parallel, delayed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import as_float_array, safe_mask, check_random_state
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as sm_formula
from utils.common_settings import *
from data_loader.data_loader_ml import DataRepo
from algorithm.base import DepressionDetectionClassifierBase
from algorithm.ml_basic import DepressionDetectionAlgorithm_ML_basic
import signal

class DepressionDetectionClassifier_ML_chikersal(DepressionDetectionClassifierBase):
    """Classifier for Chikersal et al. work. A adaboost model with gradientbossting classifier as the base model """
    def __init__(self, feature_dict, single_model_params_dict, ensemble_model_params):
        self.feature_dict = feature_dict
        self.single_model_params_dict = single_model_params_dict
        self.ensemble_model_params = ensemble_model_params
        self.feature_types = list(self.feature_dict.keys())
        
        self.single_model_dict = {k:None for k in self.feature_types}
        self.single_model_prob_dict = {k:None for k in self.feature_types}
        self.ensemble_model = None
    
    def fit(self, X, y=None):
        set_random_seed(42)
        for ft, params in self.single_model_params_dict.items():
            X_selected = X[self.feature_dict[ft]]
            if (params["model"] == "lr"):
                clf = linear_model.LogisticRegression(**{k:v for k,v in params.items() if k!="model"})
            elif (params["model"] == "gb"):
                clf = ensemble.GradientBoostingClassifier(**{k:v for k,v in params.items() if k!="model"})
            clf.fit(X_selected, y)
            self.single_model_dict[ft] = deepcopy(clf)
            prob = clf.predict_proba(X_selected)[:,1]
            self.single_model_prob_dict[ft] = deepcopy(prob)
        df_single_model_prob = pd.DataFrame(self.single_model_prob_dict)
        clf_ensemble = ensemble.AdaBoostClassifier(n_estimators = self.ensemble_model_params["n_estimators"],
                      base_estimator = ensemble.GradientBoostingClassifier(n_estimators = 50))
        clf_ensemble.fit(df_single_model_prob, y)
        self.ensemble_model = deepcopy(clf_ensemble)
    
    def predict(self, X, y=None):
        df_pred_prob = {}
        for ft, params in self.single_model_params_dict.items():
            X_selected = X[self.feature_dict[ft]]
            clf = self.single_model_dict[ft]
            prob = clf.predict_proba(X_selected)[:,1]
            df_pred_prob[ft] = deepcopy(prob)
        df_pred_prob = pd.DataFrame(df_pred_prob)
        y_pred = self.ensemble_model.predict(df_pred_prob)
        return y_pred
    
    def predict_proba(self, X, y=None):
        df_pred_prob = {}
        for ft, params in self.single_model_params_dict.items():
            X_selected = X[self.feature_dict[ft]]
            clf = self.single_model_dict[ft]
            prob = clf.predict_proba(X_selected)[:,1]
            df_pred_prob[ft] = deepcopy(prob)
        df_pred_prob = pd.DataFrame(df_pred_prob)
        y_pred_probs = self.ensemble_model.predict_proba(df_pred_prob)
        return y_pred_probs
    

class DepressionDetectionAlgorithm_ML_chikersal(DepressionDetectionAlgorithm_ML_basic):
    """Algirithm for Chikersal et al. work, extending the basic traditional ml algorithm """

    def __init__(self, config_dict = None, config_name = "ml_chikersal"):
        super().__init__(config_dict, config_name)

        if (self.flag_use_norm_features):
            self.feature_list_slope = [
                f"{f}_norm:{epoch}" for f in self.feature_list_base \
                    for epoch in ["morning", "afternoon", "evening", "night", "allday"]
            ]
            self.feature_list = [
                f"{f}_norm:14dhist" for f in self.feature_list_base
            ]
        else:
            self.feature_list_slope = [
                f"{f}:{epoch}" for f in self.feature_list_base \
                    for epoch in ["morning", "afternoon", "evening", "night", "allday"]
            ]
            self.feature_list = [
                f"{f}:14dhist" for f in self.feature_list_base
            ]

    def calc_beta(self, x: np.ndarray, y: np.ndarray):
        sum_x, sum_y = sum(x), sum(y)
        denom = len(x) * np.dot(x.T,x) - sum_x ** 2
        if (denom == 0):
            beta = np.nan
        else:
            beta = float((len(x) * np.dot(x.T,y) - sum_x * sum_y) / denom)
        return beta

    def calc_slope_list(self, df: pd.DataFrame):
        if (len(df) == 0):
            return [np.nan] * len(self.feature_list_slope)

        slope_list = []
        for f in self.feature_list_slope:
            idx_nonna = (~df[["lm_x", f]].isna()).all(axis = 1)
            if (sum(idx_nonna) == 0):
                slope_list.append(np.nan)
            else:
                x = df["lm_x"][idx_nonna].values
                y = df[f][idx_nonna].values
                slope_list.append(self.calc_beta(x,y))
        return slope_list

    def calc_bpt(self, df: pd.DataFrame):
        n_for_bic = len(df)
        k_for_bic = 3  # y = a + bx + cx' where x' = (x-bpt)*xflag i.e. x' = x-bpt when xflag = 1 and 0 otherwise
        
        # piecewise linear regression
        bpt_before_list = []
        bpt_after_list = []
        for f in self.feature_list_slope:
            minbic = 999999
            bestbpt = -1
            bestbpt_idx = None

            idx_nonna = (~df[["lm_x", f]].isna()).all(axis = 1)
            dtest = df.copy()
            dtest = dtest.rename({f:'y'}, axis = 1)
            dtest = dtest[idx_nonna]
            if (dtest.empty):
                bpt_before_list.append(np.nan)
                bpt_after_list.append(np.nan)
                continue
            for bpt in df["lm_x"]:
                dtest["lm_x2"] = np.where(dtest["lm_x"] < bpt, 0, dtest["lm_x"] - bpt)
                fpredict = sm_formula.ols(formula='y ~ lm_x + lm_x2', data=dtest).fit()
                dtest["y_pred"] = fpredict.predict()
                rss = np.linalg.norm((dtest["y"] - dtest["y_pred"]), 2)
                rss_div_by_n_for_bic = rss/n_for_bic
                if (rss_div_by_n_for_bic) != 0:
                    bic = n_for_bic * np.log(rss_div_by_n_for_bic) + k_for_bic * np.log(n_for_bic)
                else:
                    bic = 999999
                if bic < minbic:
                    minbic = bic
                    bestbpt = bpt
            if (bestbpt == -1):
                bpt_before_list.append(np.nan)
                bpt_after_list.append(np.nan)
            else:
                dtest_before = dtest[dtest["lm_x"] <= bestbpt]
                x, y = dtest_before["lm_x"].values, dtest_before["y"].values
                bpt_before_list.append(self.calc_beta(x,y))

                dtest_after = dtest[dtest["lm_x"] >= bestbpt]
                x, y = dtest_after["lm_x"].values, dtest_after["y"].values

                bpt_after_list.append(self.calc_beta(x,y))
        return bpt_before_list + bpt_after_list

    def get_mean_and_slope(self, df: pd.DataFrame):
        signal.signal(signal.SIGTERM, lambda signalNumber, frame: False)

        index_mean_wkdy = [f + f"_wkdy_mean{suffix}" for suffix in ["", "half1", "half2"] \
                                                for f in self.feature_list_slope]
        index_mean_wkend = [f + f"_wkend_mean{suffix}" for suffix in ["", "half1", "half2"] \
                                                for f in self.feature_list_slope]

        index_slope_wkdy = [f + f"_wkdy_slope{suffix}" for suffix in ["", "half1", "half2", "bpt1", "bpt2"] \
                                                for f in self.feature_list_slope]
        index_slope_wkend = [f + f"_wkend_slope{suffix}" for suffix in ["", "half1", "half2", "bpt1", "bpt2"] \
                                                for f in self.feature_list_slope]

        df_tmp_wkdy = deepcopy(df[df["date"].dt.dayofweek < 5])
        df_tmp_wkdy["wk_num"] = df_tmp_wkdy["date"].dt.week
        df_tmp_wkdy = df_tmp_wkdy.groupby("wk_num").mean().reset_index()
        if (df_tmp_wkdy.empty):
            slope_list_total_wkdy = [np.nan] * 5 * len(self.feature_list_slope)
            mean_list_total_wkdy = [np.nan] * 3 * len(self.feature_list_slope)
        else:
            df_tmp_wkdy["lm_x"] = df_tmp_wkdy["wk_num"] - df_tmp_wkdy["wk_num"].iloc[0] + 1
            df_tmp_wkdy_half1 = df_tmp_wkdy.iloc[:int(np.ceil(len(df_tmp_wkdy)/2))]
            df_tmp_wkdy_half2 = df_tmp_wkdy.iloc[int(np.ceil(len(df_tmp_wkdy)/2)):]

            mean_list_total_wkdy = list(np.concatenate([df_tmp_wkdy[self.feature_list_slope].mean().values,
                df_tmp_wkdy_half1[self.feature_list_slope].mean().values,
                df_tmp_wkdy_half2[self.feature_list_slope].mean().values],axis=0))
            slope_list_total_wkdy = self.calc_slope_list(df_tmp_wkdy) + \
                self.calc_slope_list(df_tmp_wkdy_half1) + \
                self.calc_slope_list(df_tmp_wkdy_half2)
            slope_list_total_wkdy_bpt = self.calc_bpt(df_tmp_wkdy)
            slope_list_total_wkdy += slope_list_total_wkdy_bpt


        df_tmp_wkend = deepcopy(df[df["date"].dt.dayofweek >= 5])
        df_tmp_wkend["wk_num"] = df_tmp_wkend["date"].dt.week
        df_tmp_wkend = df_tmp_wkend.groupby("wk_num").mean().reset_index()
        if (df_tmp_wkend.empty):
            slope_list_total_wkend = [np.nan] * 5 * len(self.feature_list_slope)
            mean_list_total_wkend = [np.nan] * 3 * len(self.feature_list_slope)
        else:
            df_tmp_wkend["lm_x"] = df_tmp_wkend["wk_num"] - df_tmp_wkend["wk_num"].iloc[0] + 1
            df_tmp_wkend_half1 = df_tmp_wkend.iloc[:(len(df_tmp_wkend)//2)]
            df_tmp_wkend_half2 = df_tmp_wkend.iloc[(len(df_tmp_wkend)//2):]

            mean_list_total_wkend = list(np.concatenate([df_tmp_wkend[self.feature_list_slope].mean().values,
                df_tmp_wkend_half1[self.feature_list_slope].mean().values,
                df_tmp_wkend_half2[self.feature_list_slope].mean().values],axis=0))
            slope_list_total_wkend = self.calc_slope_list(df_tmp_wkend) + \
                self.calc_slope_list(df_tmp_wkend_half1) + \
                self.calc_slope_list(df_tmp_wkend_half2)
            slope_list_total_wkend_bpt = self.calc_bpt(df_tmp_wkend)
            slope_list_total_wkend += slope_list_total_wkend_bpt
        s = pd.Series(mean_list_total_wkdy + slope_list_total_wkdy + \
                    mean_list_total_wkend + slope_list_total_wkend,
                    index = index_mean_wkdy + index_slope_wkdy + \
                            index_mean_wkend + index_slope_wkend)
        return s

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
                @globalize
                def get_mean_slope_ray(df):
                    return self.get_mean_and_slope(df)
                with Pool(NJOB) as pool:
                    X_tmp = pool.starmap(get_mean_slope_ray, [(df,) for df in df_datapoints["X_raw"]])
                
                X_tmp = pd.DataFrame(X_tmp)
                X_tmp.index = df_datapoints.index

                if (self.config["training_params"]["save_and_reload"]):
                    with open(self.save_file_path, "wb") as f:
                        pickle.dump(X_tmp, f)

        return X_tmp
    
    def prep_model(self, data_train: DataRepo, criteria: str = "f1") -> sklearn.base.ClassifierMixin:
        super().prep_model()
        set_random_seed(42)

        @ray.remote
        def train_small_cv(data_repo: DataRepo, model_parameters: dict):
            warnings.filterwarnings("ignore")
            signal.signal(signal.SIGTERM, lambda signalNumber, frame: False)
            X = data_repo.X
            y = data_repo.y
            pids = data_repo.pids

            if (model_parameters["model"] == "lr"):
                clf = linear_model.LogisticRegression(**{k:v for k,v in model_parameters.items() if k!="model"})
            elif (model_parameters["model"] == "gb"):
                clf = ensemble.GradientBoostingClassifier(**{k:v for k,v in model_parameters.items() if k!="model"})
            elif (model_parameters["model"] == "adaboost"):
                clf = ensemble.AdaBoostClassifier(n_estimators = model_parameters["n_estimators"],
                            base_estimator = ensemble.GradientBoostingClassifier(n_estimators = 50))
            else:
                raise RuntimeError("Unknown model type")
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            r = cross_validate(clf, X=X, y=y, groups= pids, cv = cv,
                    scoring = utils_ml.results_report_sklearn, return_train_score=False)
            r = {k:np.mean(v) for k,v in r.items()}
            r.update({"parameters":model_parameters})
            return r

        temporal_slice = {}
        for feat in data_train.X.columns:
            sp = feat.split(":")
            feat_type = sp[0]
            if (feat_type not in temporal_slice):
                temporal_slice[feat_type] = {}
            slice_str = sp[-1]
            epoch, wk, comp = slice_str.split("_")
            k = f"{wk}_{epoch}"
            if (k in temporal_slice[feat_type]):
                temporal_slice[feat_type][k] += [feat]
            else:
                temporal_slice[feat_type][k] = [feat]

        selected_features_dict = {}
        for ft, d in temporal_slice.items():
            selected_features_tmp = []
            for epoch, features in d.items():
                if (len(features) == 0):
                    continue
                X_feat = data_train.X[features]
                rlr = RandomizedLogisticRegression(random_state = 42)
                rlr.fit(X_feat, data_train.y)
                selected_features_tmp += list(np.array(features)[np.where(rlr.scores_ > np.quantile(rlr.scores_, 0.5))[0]])
            if len(selected_features_tmp) == 0:
                continue
            else:
                X_feat_selected = data_train.X[selected_features_tmp]
                rlr = RandomizedLogisticRegression(random_state = 42)
                rlr.fit(X_feat_selected, data_train.y)
                selected_features_final = list(np.array(selected_features_tmp)[np.where(rlr.scores_ > np.quantile(rlr.scores_, 0.5))[0]])
                selected_features_dict[ft] = selected_features_final

        # Each single model Part
        parameters_list = []
        C_list = [10**i for i in range(-4,5)]
        for C in C_list:
            parameters_tmp = {"penalty":"l1","max_iter":50000, "solver":"saga","C":C, "model":"lr", "random_state":42}
            parameters_list.append(parameters_tmp)
        N_list = [4*i for i in range(1,11)]
        for N in N_list:
            parameters_tmp = {"n_estimators":N, "model":"gb", "random_state":42}
            parameters_list.append(parameters_tmp)

        best_params_single_dict = {}
        for ft, features in selected_features_dict.items():
            data_train_ft = deepcopy(data_train)
            data_train_ft.X = data_train_ft.X[features]
            
            data_train_ft_id = ray.put(data_train_ft)
            results_list = ray.get([train_small_cv.remote(data_train_ft_id,i) for i in parameters_list])

            results_list = pd.DataFrame(results_list)

            best_row = results_list.iloc[results_list[f"test_{criteria}"].argmax()]
            best_params = best_row['parameters']
            best_params_single_dict[ft] = best_params
            if (self.verbose > 0):
                print(ft, best_row, best_params)
                
        # Ensemble Part
        model_dict = {}
        prob_dict = {}
        for ft, params in best_params_single_dict.items():
            if (params["model"] == "lr"):
                clf = linear_model.LogisticRegression(**{k:v for k,v in params.items() if k!="model"})
            elif (params["model"] == "gb"):
                clf = ensemble.GradientBoostingClassifier(**{k:v for k,v in params.items() if k!="model"})
            clf.fit(data_train.X[selected_features_dict[ft]], data_train.y)
            model_dict[ft] = deepcopy(clf)
            prob = clf.predict_proba(data_train.X[selected_features_dict[ft]])[:,1]
            prob_dict[ft] = deepcopy(prob)

        data_train_ensemble = DataRepo(X=pd.DataFrame(prob_dict).values, y=data_train.y, pids=data_train.pids)

        parameters_list_ensemble = []
        N_list = [10*i for i in range(1,11)]
        for N in N_list:
            parameters_tmp = {"n_estimators":N, "model":"adaboost", "random_state":42}
            parameters_list_ensemble.append(parameters_tmp)

        results_list = []
        data_train_ensemble_id = ray.put(data_train_ensemble)
        results_list = ray.get([train_small_cv.remote(data_train_ensemble_id,i) for i in parameters_list_ensemble])

        results_list = pd.DataFrame(results_list)

        best_row = results_list.iloc[results_list[f"test_{criteria}"].argmax()]
        best_params_ensemble = best_row['parameters']
        if (self.verbose > 0):
            print(best_row, best_params_ensemble)
            
        clf = DepressionDetectionClassifier_ML_chikersal(feature_dict=selected_features_dict,
                                single_model_params_dict=best_params_single_dict,
                                ensemble_model_params=best_params_ensemble)
        return clf

""" 
    Modifed from Sklearn 0.15 RandomizedLogisticRegression
    https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.RandomizedLogisticRegression.html
"""

def _assert_all_finite(X):
    """Throw a ValueError if X contains NaN or infinity."""
    X = np.asanyarray(X)
    s = "AllFloat"
    if (X.dtype.char in np.typecodes[s] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)

def safe_asarray(X, dtype=None, order=None, copy=False, force_all_finite=True):
    """Convert X to an array or CSC/CSR/COO sparse matrix.
    Prevents copying X when possible. Sparse matrices in CSR, CSC and COO
    formats are passed through. Other sparse formats are converted to CSR
    (somewhat arbitrarily).
    If a specific compressed sparse format is required, use atleast2d_or_cs{c,r}
    instead.
    """
    if sp.issparse(X):
        if not isinstance(X, (sp.coo_matrix, sp.csc_matrix, sp.csr_matrix)):
            X = X.tocsr()
        elif copy:
            X = X.copy()
        if force_all_finite:
            _assert_all_finite(X.data)
        # enforces dtype on data array (order should be kept the same).
        X.data = np.asarray(X.data, dtype=dtype)
    else:
        X = np.array(X, dtype=dtype, order=order, copy=copy)
        if force_all_finite:
            _assert_all_finite(X)
    return X


def center_data(X, y, fit_intercept, normalize=False, copy=True,
                sample_weight=None):
    """
    Centers data to have mean zero along axis 0. This is here because
    nearly all linear models will want their data to be centered.
    If sample_weight is not None, then the weighted mean of X and y
    is zero, and not the mean itself
    """
    X = as_float_array(X, copy=copy)
    if fit_intercept:
        if isinstance(sample_weight, numbers.Number):
            sample_weight = None
        if sp.issparse(X):
            X_mean = np.zeros(X.shape[1])
            X_std = np.ones(X.shape[1])
        else:
            X_mean = np.average(X, axis=0, weights=sample_weight)
            X -= X_mean
            if normalize:
                # XXX: currently scaled to variance=n_samples
                X_std = np.sqrt(np.sum(X ** 2, axis=0))
                X_std[X_std == 0] = 1
                X /= X_std
            else:
                X_std = np.ones(X.shape[1])
        y_mean = np.average(y, axis=0, weights=sample_weight)
        y = y - y_mean
    else:
        X_mean = np.zeros(X.shape[1])
        X_std = np.ones(X.shape[1])
        y_mean = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)
    return X, y, X_mean, y_mean, X_std

def _resample_model(estimator_func, X, y, scaling=.5, n_resampling=200,
                    n_jobs=1, verbose=False, pre_dispatch='3*n_jobs',
                    random_state=None, sample_fraction=.75, **params):
    random_state = check_random_state(random_state)
    # We are generating 1 - weights, and not weights
    n_samples, n_features = X.shape

    if not (0 < scaling < 1):
        raise ValueError(
            "'scaling' should be between 0 and 1. Got %r instead." % scaling)

    scaling = 1. - scaling
    scores_ = 0.0

    mask_tmp = random_state.rand(n_samples) < sample_fraction
    while (len(set(y[mask_tmp])) < 2):
        mask_tmp = random_state.rand(n_samples) < sample_fraction

    for active_set in Parallel(n_jobs=n_jobs, verbose=verbose,
                               pre_dispatch=pre_dispatch)(
            delayed(estimator_func)(
                X, y, weights=scaling * random_state.randint(
                    0, 2, size=(n_features,)),
                mask=mask_tmp,
                verbose=max(0, verbose - 1),
                **params)
            for _ in range(n_resampling)):
        scores_ += active_set

    scores_ /= n_resampling
    return scores_


class BaseRandomizedLinearModel(six.with_metaclass(ABCMeta, BaseEstimator,
                                                   TransformerMixin)):
    """Base class to implement randomized linear models for feature selection
    This implements the strategy by Meinshausen and Buhlman:
    stability selection with randomized sampling, and random re-weighting of
    the penalty.
    """

    @abstractmethod
    def __init__(self):
        pass

    _center_data = staticmethod(center_data)

    def fit(self, X, y):
        """Fit the model using X, y as training data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            training data.
        y : array-like, shape = [n_samples]
            target values.
        Returns
        -------
        self : object
            returns an instance of self.
        """
        X, y = check_arrays(X, y)
        X = as_float_array(X, copy=False)
        n_samples, n_features = X.shape

        X, y, X_mean, y_mean, X_std = self._center_data(X, y,
                                                        self.fit_intercept,
                                                        self.normalize)

        estimator_func, params = self._make_estimator_and_params(X, y)
        memory = self.memory
        if isinstance(memory, six.string_types):
            memory = Memory(cachedir=memory)

        scores_ = memory.cache(
            _resample_model, ignore=['verbose', 'n_jobs', 'pre_dispatch']
        )(
            estimator_func, X, y,
            scaling=self.scaling, n_resampling=self.n_resampling,
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=self.pre_dispatch, random_state=self.random_state,
            sample_fraction=self.sample_fraction, **params)

        if scores_.ndim == 1:
            scores_ = scores_[:, np.newaxis]
        self.all_scores_ = scores_
        self.scores_ = np.max(self.all_scores_, axis=1)
        return self

    def _make_estimator_and_params(self, X, y):
        """Return the parameters passed to the estimator"""
        raise NotImplementedError

    def get_support(self, indices=False):
        """Return a mask, or list, of the features/indices selected."""
        mask = self.scores_ > self.selection_threshold
        return mask if not indices else np.where(mask)[0]

    def transform(self, X):
        """Transform a new matrix using the selected features"""
        mask = self.get_support()
        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return safe_asarray(X)[:, safe_mask(X, mask)]

    def inverse_transform(self, X):
        """Transform a new matrix using the selected features"""
        support = self.get_support()
        if X.ndim == 1:
            X = X[None, :]
        Xt = np.zeros((X.shape[0], support.size))
        Xt[:, support] = X
        return Xt


def _randomized_logistic(X, y, weights, mask, C=1., verbose=False,
                         fit_intercept=True, tol=1e-3):
    X = X[safe_mask(X, mask)]
    y = y[mask]
    if sp.issparse(X):
        size = len(weights)
        weight_dia = sp.dia_matrix((1 - weights, 0), (size, size))
        X = X * weight_dia
    else:
        X *= (1 - weights)

    C = np.atleast_1d(np.asarray(C, dtype=np.float))
    scores = np.zeros((X.shape[1], len(C)), dtype=np.bool)

    for this_C, this_scores in zip(C, scores.T):
        # TODO : would be great to do it with a warm_start ...
        clf = LogisticRegression(C=this_C, tol=tol, penalty='l1', dual=False, solver = "liblinear",
                                 fit_intercept=fit_intercept)
        clf.fit(X, y)
        this_scores[:] = np.any(
            np.abs(clf.coef_) > 10 * np.finfo(np.float).eps, axis=0)
    return scores


class RandomizedLogisticRegression(BaseRandomizedLinearModel):
    """Randomized Logistic Regression
    Randomized Regression works by resampling the train data and computing
    a LogisticRegression on each resampling. In short, the features selected
    more often are good features. It is also known as stability selection.
    Parameters
    ----------
    C : float, optional, default=1
        The regularization parameter C in the LogisticRegression.
    scaling : float, optional, default=0.5
        The alpha parameter in the stability selection article used to
        randomly scale the features. Should be between 0 and 1.
    sample_fraction : float, optional, default=0.75
        The fraction of samples to be used in each randomized design.
        Should be between 0 and 1. If 1, all samples are used.
    n_resampling : int, optional, default=200
        Number of randomized models.
    selection_threshold: float, optional, default=0.25
        The score above which features should be selected.
    fit_intercept : boolean, optional, default=True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    verbose : boolean or integer, optional
        Sets the verbosity amount
    normalize : boolean, optional, default=True
        If True, the regressors X will be normalized before regression.
    tol : float, optional, default=1e-3
         tolerance for stopping criteria of LogisticRegression
    n_jobs : integer, optional
        Number of CPUs to use during the resampling. If '-1', use
        all the CPUs
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    memory : Instance of joblib.Memory or string
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.
    Attributes
    ----------
    `scores_` : array, shape = [n_features]
        Feature scores between 0 and 1.
    `all_scores_` : array, shape = [n_features, n_reg_parameter]
        Feature scores between 0 and 1 for all values of the regularization \
        parameter. The reference article suggests ``scores_`` is the max \
        of ``all_scores_``.
    Examples
    --------
    >>> from sklearn.linear_model import RandomizedLogisticRegression
    >>> randomized_logistic = RandomizedLogisticRegression()
    Notes
    -----
    See examples/linear_model/plot_sparse_recovery.py for an example.
    References
    ----------
    Stability selection
    Nicolai Meinshausen, Peter Buhlmann
    Journal of the Royal Statistical Society: Series B
    Volume 72, Issue 4, pages 417-473, September 2010
    DOI: 10.1111/j.1467-9868.2010.00740.x
    See also
    --------
    RandomizedLasso, Lasso, ElasticNet
    """
    def __init__(self, C=1, scaling=.5, sample_fraction=.75,
                 n_resampling=200,
                 selection_threshold=.25, tol=1e-3,
                 fit_intercept=True, verbose=False,
                 normalize=True,
                 random_state=None,
                 n_jobs=1, pre_dispatch='3*n_jobs',
                 memory=Memory(cachedir=None, verbose=0)):
        self.C = C
        self.scaling = scaling
        self.sample_fraction = sample_fraction
        self.n_resampling = n_resampling
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.normalize = normalize
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.selection_threshold = selection_threshold
        self.pre_dispatch = pre_dispatch
        self.memory = memory

    def _make_estimator_and_params(self, X, y):
        params = dict(C=self.C, tol=self.tol,
                      fit_intercept=self.fit_intercept)
        return _randomized_logistic, params

    def _center_data(self, X, y, fit_intercept, normalize=False):
        """Center the data in X but not in y"""
        X, _, Xmean, _, X_std = center_data(X, y, fit_intercept,
                                            normalize=normalize)
        return X, y, Xmean, y, X_std


def _num_samples(x):
    """Return number of samples in array-like x."""
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %r" % x)
    return x.shape[0] if hasattr(x, 'shape') else len(x)

def check_arrays(*arrays, **options):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    By default lists and tuples are converted to numpy arrays.
    It is possible to enforce certain properties, such as dtype, continguity
    and sparse matrix format (if a sparse matrix is passed).
    Converting lists to arrays can be disabled by setting ``allow_lists=True``.
    Lists can then contain arbitrary objects and are not checked for dtype,
    finiteness or anything else but length. Arrays are still checked
    and possibly converted.
    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]
        Python lists or tuples occurring in arrays are converted to 1D numpy
        arrays, unless allow_lists is specified.
    sparse_format : 'csr', 'csc' or 'dense', None by default
        If not None, any scipy.sparse matrix is converted to
        Compressed Sparse Rows or Compressed Sparse Columns representations.
        If 'dense', an error is raised when a sparse array is
        passed.
    copy : boolean, False by default
        If copy is True, ensure that returned arrays are copies of the original
        (if not already converted to another format earlier in the process).
    check_ccontiguous : boolean, False by default
        Check that the arrays are C contiguous
    dtype : a numpy dtype instance, None by default
        Enforce a specific dtype.
    allow_lists : bool
        Allow lists of arbitrary objects as input, just check their length.
        Disables
    allow_nans : boolean, False by default
        Allows nans in the arrays
    allow_nd : boolean, False by default
        Allows arrays of more than 2 dimensions.
    """
    sparse_format = options.pop('sparse_format', None)
    if sparse_format not in (None, 'csr', 'csc', 'dense'):
        raise ValueError('Unexpected sparse format: %r' % sparse_format)
    copy = options.pop('copy', False)
    check_ccontiguous = options.pop('check_ccontiguous', False)
    dtype = options.pop('dtype', None)
    allow_lists = options.pop('allow_lists', False)
    allow_nans = options.pop('allow_nans', False)
    allow_nd = options.pop('allow_nd', False)

    if options:
        raise TypeError("Unexpected keyword arguments: %r" % options.keys())

    if len(arrays) == 0:
        return None

    n_samples = _num_samples(arrays[0])

    checked_arrays = []
    for array in arrays:
        array_orig = array
        if array is None:
            # special case: ignore optional y=None kwarg pattern
            checked_arrays.append(array)
            continue
        size = _num_samples(array)

        if size != n_samples:
            raise ValueError("Found array with dim %d. Expected %d"
                             % (size, n_samples))

        if not allow_lists or hasattr(array, "shape"):
            if sp.issparse(array):
                if sparse_format == 'csr':
                    array = array.tocsr()
                elif sparse_format == 'csc':
                    array = array.tocsc()
                elif sparse_format == 'dense':
                    raise TypeError('A sparse matrix was passed, but dense '
                                    'data is required. Use X.toarray() to '
                                    'convert to a dense numpy array.')
                if check_ccontiguous:
                    array.data = np.ascontiguousarray(array.data, dtype=dtype)
                elif hasattr(array, 'data'):
                    array.data = np.asarray(array.data, dtype=dtype)
                elif array.dtype != dtype:
                    array = array.astype(dtype)
                if not allow_nans:
                    if hasattr(array, 'data'):
                        _assert_all_finite(array.data)
                    else:
                        _assert_all_finite(array.values())
            else:
                if check_ccontiguous:
                    array = np.ascontiguousarray(array, dtype=dtype)
                else:
                    array = np.asarray(array, dtype=dtype)
                if not allow_nans:
                    _assert_all_finite(array)

            if not allow_nd and array.ndim >= 3:
                raise ValueError("Found array with dim %d. Expected <= 2" %
                                 array.ndim)

        if copy and array is array_orig:
            array = array.copy()
        checked_arrays.append(array)

    return checked_arrays