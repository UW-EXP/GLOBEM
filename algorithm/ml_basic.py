import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.base import DepressionDetectionAlgorithmBase
from data_loader.data_loader_ml import DatasetDict, DataRepo
from utils import path_definitions

class DepressionDetectionAlgorithm_ML_basic(DepressionDetectionAlgorithmBase):
    """Template for traditional machine learning based depression deetection model object"""
    def __init__(self, config_dict:dict = None, config_name:str = "canzian"):
        """Init function by defining config

        Args:
            config_dict (dict, optional): a dictionary of config parameters. Defaults to None.
            config_name (str, optional): the name of config file in "config" folder. Defaults to "canzian".
        """
        super().__init__()

        if (config_dict is None):
            with open(os.path.join(path_definitions.CONFIG_PATH, f"{config_name}.yaml"), "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_dict

        with open(os.path.join(path_definitions.CONFIG_PATH, f"global_config.yaml"), "r") as f:
            self.global_config = yaml.safe_load(f)

        # define the feature list
        self.flag_more_feat_types = self.global_config["all"].get("flag_more_feat_types", False)
        if (self.flag_more_feat_types):
            self.feature_list = self.config["feature_definition"].get("feature_list_more_feat_types", None)
            self.feature_list_base = self.config["feature_definition"].get("feature_list_more_feat_types_base", None)
        else:
            self.feature_list = self.config["feature_definition"].get("feature_list", None)
            self.feature_list_base = self.config["feature_definition"].get("feature_list_base", None)

        self.flag_use_norm_features = self.config["feature_definition"]["use_norm_features"]
        self.verbose = self.config["training_params"]["verbose"]

        if self.flag_use_norm_features:
            feature_list_new = []
            for idx, f in enumerate(self.feature_list):
                ft, fn, seg = f.split(":")
                new_f = f"{ft}:{fn}_norm:{seg}"
                feature_list_new.append(new_f)
            self.feature_list = feature_list_new

        # if there is global_config, override some params
        if ("training_params" in self.global_config['ml']):
            if ("save_and_reload" in self.global_config['ml']["training_params"]):
                self.config["training_params"]["save_and_reload"] = \
                    self.global_config["ml"]["training_params"]["save_and_reload"]

        # Reload feature pkl files that may have been computed and saved
        self.results_save_folder = os.path.join(path_definitions.TMP_PATH, self.config["name"])
        Path(self.results_save_folder).mkdir(parents=True, exist_ok=True)
        save_files = glob.glob(os.path.join(self.results_save_folder, "*.pkl"))
        self.results_reload_dict = {}
        for save_file in save_files:
            key_str = os.path.basename(save_file).replace(".pkl","")
            key, pred_target_str = key_str.split("--")
            if pred_target_str not in self.results_reload_dict:
                self.results_reload_dict[pred_target_str] = {}
            with open(save_file , "rb") as f:
                self.results_reload_dict[pred_target_str][key] = pickle.load(f)

    def prep_X_reload(self, key: str, pred_target: str) -> Tuple[bool, pd.DataFrame]:
        """Reload feature matrix if the individual dataset has been covered in self.results_reload_dict

        Args:
            key (str): dataset key ds_key

        Returns:
            Tuple[bool, pd.DataFrame]: (whether data is reloaded, reloaded dataframe - can be None)
        """
        if not self.config["training_params"]["save_and_reload"]:
            return False, None
        if pred_target not in self.results_reload_dict:
            return False, None 
        key_list = key.split(":")
        flag_seperate_reload = set(key_list).issubset(set(self.results_reload_dict[pred_target].keys()))
        if (flag_seperate_reload):
            X_tmp_list = []
            for key in key_list:
                print("reload, ", key)
                X_tmp_list.append(self.results_reload_dict[pred_target][key])
            X_tmp_reload = pd.concat(X_tmp_list).reset_index(drop=True)
        else:
            X_tmp_reload = None
        return flag_seperate_reload, X_tmp_reload

    def prep_X(self, key:str, pred_target:str, df_datapoints: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix

        Args:
            key (str): dataset key ds_key
            pred_target (str): dataset prediction target
            df_datapoints (pd.DataFrame): data frame of slices datapoints

        Returns:
            pd.DataFrame: calculated features
        """
        self.save_file_path = os.path.join(self.results_save_folder, key + "--" + pred_target + ".pkl")

        # if the config indicate reloading, and the feature data has been saved. Skip the computation
        if (self.config["training_params"]["save_and_reload"] and os.path.exists(self.save_file_path)):
            with open(self.save_file_path, "rb") as f:
                X_tmp = pickle.load(f)
        else:
            # if the config indicate reloading, but the feature data has not been saved. Try reload individual dataset
            flag_seperate_reload, X_tmp_reload = self.prep_X_reload(key, pred_target)
            if flag_seperate_reload:
                X_tmp = X_tmp_reload
            else:
                # in the basic version, it takes the last value
                X_tmp = df_datapoints["X_raw"].apply(lambda x :
                    pd.Series(data = x[self.feature_list].iloc[-1].values, index = self.feature_list).T)

                # save the obj to save time for future computation
                if (self.config["training_params"]["save_and_reload"]):
                    with open(self.save_file_path, "wb") as f:
                        pickle.dump(X_tmp, f)
        return X_tmp

    def prep_data_repo(self, dataset:DatasetDict, flag_train:bool = True) -> DataRepo:
        """Prepare the DataRepo object given the DatasetDict object.
            Unique in each algorithm.

        Args:
            dataset (DatasetDict): target dataset
            flag_train (bool, optional): whether this is the train set.
                If it is, it will perform feature filtering to remove very empty features.
                If it is not, it will skip feature filtering to maximize compatibility. Defaults to True.

        Returns:
            DataRepo: a prepared data repo
        """
        super().prep_data_repo(flag_train = flag_train)
        set_random_seed(42)

        df_datapoints = deepcopy(dataset.datapoints)

        X_tmp = self.prep_X(key=dataset.key, pred_target=dataset.prediction_target, df_datapoints=df_datapoints)

        # filter
        shape1 = X_tmp.shape
        if (flag_train): # Testing dataset does not delete features to ensure compatibility
            X_tmp = X_tmp[X_tmp.columns[(X_tmp.isna().sum(axis = 0) < \
                (X_tmp.shape[0] * self.config["feature_definition"]["empty_feature_filtering_th"]))]] # filter very empty features
        shape2 = X_tmp.shape
        del_cols = shape1[1] - shape2[1]
        if (self.config["feature_definition"]["empty_feature_filtering_th"] > 0.8):
            assert del_cols == 0
        X_tmp = X_tmp[X_tmp.isna().sum(axis = 1) < X_tmp.shape[1] / 2]  # filter very empty days
        shape3 = X_tmp.shape
        del_rows = shape2[0] - shape3[0]

        X = deepcopy(X_tmp)

        scl = RobustScaler(quantile_range = (5,95), unit_variance = True).fit(X)
        X[X.columns] = scl.transform(X)
        
        if (self.verbose > 0):
            print(f"filter {del_cols} cols and {del_rows} rows")
            print(f"NA rate: {100* X.isna().sum().sum() / X.shape[0] / X.shape[1]}%" )

        X = X.fillna(X.mean())
        X = X.fillna(0) # for those completely empty features (e.g., one dataset does not have the feature)

        y = df_datapoints["y_raw"].loc[X.index]
        pids = df_datapoints["pid"].loc[X.index]
        device_types = df_datapoints["device_type"].loc[X.index]

        if (self.config["feature_definition"]["include_device_type"]):
            X["device_type"] = device_types
            X["device_type"] = X["device_type"].apply(lambda x : 1 if x == "ios" else 0)

        self.data_repo = DataRepo(X=X, y=y, pids=pids)
        return self.data_repo

    def prep_model(self):
        """ To be tailored by each algorithm. """
        pass
