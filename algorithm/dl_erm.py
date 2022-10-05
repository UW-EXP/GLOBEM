"""
Implementation of Empirical Risk Minimization (ERM) algorithm for depression detection

reference:

Vladimir N Vapnik. 1999. An overview of statistical learning theory.
IEEE transactions on neural networks 10, 5 (1999), 988–999.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.base import DepressionDetectionAlgorithmBase
from data_loader.data_loader_dl import data_loader_np, prep_repo_np_dict_feature_prep, MultiSourceDataGenerator, dl_feat_preparation
from data_loader.data_loader_ml import DatasetDict, DataRepo, DataRepo_tf
from algorithm.base import DepressionDetectionClassifierBase
from utils import network, path_definitions
from tensorflow.python.data.ops.dataset_ops import FlatMapDataset

class DepressionDetectionClassifier_DL_erm(DepressionDetectionClassifierBase):
    """Basic ERM-based deep learning algorithm. Other dl classifier can extend this class"""
    def __init__(self, config):
        self.config = config
        self.model_params = self.config["model_params"]
        self.training_params = self.config["training_params"]

        # define network architecture
        if (self.model_params["arch"] == "1dCNN"):
            self.clf = network.build_1dCNN(**self.model_params)
        elif (self.model_params["arch"] == "2dCNN"):
            self.clf = network.build_2dCNN(**self.model_params)
        elif (self.model_params["arch"] == "LSTM"):
            self.clf = network.build_LSTM(**self.model_params)
        elif (self.model_params["arch"] == "Transformer"):
            self.clf = network.build_Transformer(**self.model_params)
        
        # define results saving folder
        folder_name = self.config["data_loader"]["training_dataset_key"]
        self.results_save_folder = os.path.join(path_definitions.TMP_PATH, self.config["name"],
            "input_shape-" + "_".join([str(i) for i in self.model_params["input_shape"]]),
            self.config["training_params"]["eval_task"],
            folder_name,
        )
        Path(self.results_save_folder).mkdir(parents=True, exist_ok=True)

    def __assert__(self, X):
        """ Check whether input is expected """
        if (type(X) is dict):
            # if a dictionary, each value should be a tf dataset
            self.flag_X_dict = True
            for k, v in X.items():
                assert type(v) is FlatMapDataset
        else:
            # if not a directionary, itself should be tf dataset
            self.flag_X_dict = False
            assert type(X) is FlatMapDataset

    def prep_eval_callbacks(self, X):
        """ Callback function for evaluation on train/val/test set """
        ds_train = X["val_whole"]
        if "test" in X:
            ds_test = X["test"]
        elif "val" in X:
            ds_test = X["val"]
        else:
            ds_test = None
        return network.EvaluationBasicCallback(model_obj=self,
                dataset_train=ds_train, dataset_test=ds_test,
                interval=1, verbose=self.training_params["verbose"])

    def prep_callbacks(self, X):
        """ Define callback functions. Can be extended by other algorithms. """
        self.earlystopping = EarlyStopping(
            monitor="val_loss", mode="min",
            patience=1000, verbose=0,
        )

        self.model_saver = network.ModelMemoryCheckpoint()

        self.tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress = False)

        # The purpose is to evaluate each step's performance
        if (type(X) is dict):
            assert "val_whole" in X
            self.eval_callback = self.prep_eval_callbacks(X)
            return [self.earlystopping, self.model_saver, self.eval_callback, self.tqdm_callback]
        else:
            self.eval_callback = None
            return [self.earlystopping, self.model_saver, self.tqdm_callback]

    def find_best_epoch_direct(self, df_results_record, metrics = "val_loss", stop_idx = 1000):
        """ Pick the best epoch based on the train/val results """
        df_results_record_buf = df_results_record.reset_index(drop=True)
        if ("loss" in metrics):
            epoch_idx = df_results_record_buf.iloc[:stop_idx][metrics].idxmin()
        else:
            epoch_idx = df_results_record_buf.iloc[:stop_idx][metrics].idxmax()
        if (pd.isna(epoch_idx)):
            epoch_idx = -1
        return df_results_record_buf.iloc[epoch_idx]["epoch"]

    def find_best_epoch_on_test(self, df_results_record, metrics = "logs_val_loss",
            test_metrics_ = "balanced_acc_test", stop_idx = 1000):
        """ Pick the best epoch based on the test results - involve a bit information leakage """
        df_buf = deepcopy(df_results_record)
        flag_metrics_flip = -1 if "loss" in metrics else 1 # if loss, will look for minimum
        if (test_metrics_ in df_buf):
            test_metrics = test_metrics_
            test_cfmtx = "cfmtx_test"
        else:
            test_metrics = "balanced_acc_train"
            test_cfmtx = "cfmtx_train"
        best_metric_train_list = []
        best_metric_test_list = []
        best_metric_test_cfmtx_list = []

        best_metric_train = -9999
        current_metric_test = 0

        best_metric_train = flag_metrics_flip * df_buf.iloc[0][metrics]
        current_metric_test = df_buf.iloc[0][test_metrics]
        current_metric_test_cfmtx = df_buf.iloc[0][test_cfmtx]
        
        for idx, row in df_buf.iterrows():
            if ((flag_metrics_flip * row[metrics]) > best_metric_train):
                best_metric_train = flag_metrics_flip * row[metrics]
                current_metric_test = row[test_metrics]
                current_metric_test_cfmtx = row[test_cfmtx]
            best_metric_train_list.append(best_metric_train)
            best_metric_test_list.append(current_metric_test)
            best_metric_test_cfmtx_list.append(current_metric_test_cfmtx)
        df_buf[f"best_tune_train:{metrics}"] = best_metric_train_list
        df_buf[f"best_tune:{test_metrics}"] = best_metric_test_list
        df_buf[f"best_tune:{test_cfmtx}"] = best_metric_test_cfmtx_list
        return df_buf.iloc[df_buf.iloc[:stop_idx][f"best_tune:{test_metrics}"].argmax()]["epoch"]

    def save_epoch_results(self, df_results_record):
        """ Save evalution results """
        save_obj = {
            "df_results_record": df_results_record,
            "model_repo_dict": self.model_saver.model_repo_dict
        }
        if "clustering" not in self.config["name"]:
            file_path = os.path.join(self.results_save_folder, "results_saving.pkl")
        else:
            file_path = os.path.join(self.results_save_folder, f"results_saving_c{self.current_cluster_idx}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(save_obj, f)
    
    def find_best_epoch(self):
        """ Find the best epoch. The specific strategy is determined by the config file.
            Default it will be based on the train/val set"""
        if (self.eval_callback is not None):
            df_results_record = pd.DataFrame(self.eval_callback.results_record)
            if (self.training_params["best_epoch_strategy"] == "direct"):
                best_epoch = int(self.find_best_epoch_direct(df_results_record,
                    metrics = "balanced_acc_train" if ("balanced_acc_train" in df_results_record and
                                        not pd.isna(df_results_record["balanced_acc_train"]).any())
                        else "logs_loss"))
            elif (self.training_params["best_epoch_strategy"] == "on_test"):
                best_epoch = int(self.find_best_epoch_on_test(df_results_record,
                    metrics = "balanced_acc_train" if ("balanced_acc_train" in df_results_record and
                                        not pd.isna(df_results_record["balanced_acc_train"]).any())
                        else "logs_loss"))
        else:
            df_results_record = pd.DataFrame(self.log_history)
            df_results_record.columns = ["logs_" + c for c in df_results_record.columns]
            df_results_record["epoch"] = np.arange(len(df_results_record))
            best_epoch = int(self.find_best_epoch_direct(df_results_record, metrics = "logs_loss"))

        if (self.training_params["verbose"] > 0):
            print("best epoch ", best_epoch)
        
        self.save_epoch_results(df_results_record)

        return best_epoch, df_results_record

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(42)

        self.__assert__(X)

        model_optimizer = network.prep_model_optimizer(self.training_params)

        self.clf.compile(loss = 'categorical_crossentropy', optimizer = model_optimizer, metrics="acc")

        callbacks = self.prep_callbacks(X)

        # if the skip training is turned off, just do the regular training
        if (self.training_params.get("skip_training", False) == False):
            history = self.clf.fit(x = X["train"] if self.flag_X_dict else X,
                    steps_per_epoch = self.training_params["steps_per_epoch"],
                    epochs = self.training_params["epochs"],
                    validation_data = X["val"] if self.flag_X_dict else X,
                    verbose = 1 if self.training_params["verbose"] > 1 else 0,
                    callbacks = callbacks
                    )
            self.log_history = history.history

            best_epoch, df_results_record = self.find_best_epoch()

            self.clf.set_weights(self.model_saver.model_repo_dict[best_epoch])
        else: # otherwise, load the saving results to avoid the re-training. Use it carefully!
            df_results_record = self.fit_skip_training()

        return df_results_record

    def fit_skip_training(self):
        """ Only applicable when the model has been trained and the results have been saved. """

        # dl models except clustering as its intermediate results will be saved separately for each cluster
        if "clustering" not in self.config["name"]:
            file_path = os.path.join(self.results_save_folder, "results_saving.pkl")
        else:
            file_path = os.path.join(self.results_save_folder, f"results_saving_c{self.current_cluster_idx}.pkl")
        with open(file_path, "rb") as f:
            save_obj = pickle.load(f)
        df_results_record = save_obj["df_results_record"]
        self.model_saver = network.ModelMemoryCheckpoint()
        self.model_saver.model_repo_dict = save_obj["model_repo_dict"]

        if (self.training_params["best_epoch_strategy"] == "direct"):
            best_epoch = int(self.find_best_epoch_direct(df_results_record,
                metrics = "balanced_acc_train" if ("balanced_acc_train" in df_results_record and
                                    not pd.isna(df_results_record["balanced_acc_train"]).any())
                    else "logs_loss"))
        elif (self.training_params["best_epoch_strategy"] == "on_test"):
            best_epoch = int(self.find_best_epoch_on_test(df_results_record,
                metrics = "balanced_acc_train" if ("balanced_acc_train" in df_results_record and
                                    not pd.isna(df_results_record["balanced_acc_train"]).any())
                    else "logs_loss"))
        if (self.training_params["verbose"] > 0):
            print("skip training, best epoch", best_epoch)
        self.clf.set_weights(self.model_saver.model_repo_dict[best_epoch])
        return df_results_record

    def predict(self, X, y=None):
        self.__assert__(X)
        if (self.flag_X_dict):
            X_ = X["val_whole"] # only use the whole val set for eval
        else:
            X_ = X
        for data, label in X_:
            return np.argmax(self.clf.predict(data), axis = 1)

    def predict_proba(self, X, y=None):
        self.__assert__(X)
        if (self.flag_X_dict):
            X_ = X["val_whole"] # only use the whole val set for eval
        else:
            X_ = X
        for data, label in X_:
            return self.clf.predict(data)

class DepressionDetectionAlgorithm_DL_erm(DepressionDetectionAlgorithmBase):
    """Basic deep-learning based depression detection algorithm. Other dl algorithms can extend this class"""

    def __init__(self, config_dict = None, config_name = "dl_erm_1dcnn"):
        
        super().__init__()

        if (config_dict is None):
            with open(os.path.join(path_definitions.CONFIG_PATH, f"{config_name}.yaml"), "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_dict

        ds_keys_dict = {
            pred_target: global_config["all"]["ds_keys"] for pred_target in global_config["all"]["prediction_tasks"]
        }

        # DL models will preload the np pkl files. This is a different structure of traditional model pipeline.
        self.data_repo_np_dict = data_loader_np(ds_keys_dict=ds_keys_dict,
            flag_normalize=self.config["feature_definition"]["use_norm_features"],
            flag_more_feat_types=global_config["all"]["flag_more_feat_types"])
        feat_prep = dl_feat_preparation(config_name="dl_feat_prep",
            flag_more_feat_types=global_config["all"]["flag_more_feat_types"])
        self.data_repo_np_dict = prep_repo_np_dict_feature_prep(self.data_repo_np_dict,
            selected_feature_idx=feat_prep.selected_feature_idx, ndim=self.config["model_params"]["input_dim"])
        self.input_shape_dict = {}
        for pt, data_repo_np_dict_pt in self.data_repo_np_dict.items():
            self.input_shape_dict[pt] = {}
            ds_keys = list(data_repo_np_dict_pt.keys())
            for ds, data_repo_np_ptds in data_repo_np_dict_pt.items():
                self.input_shape_dict[pt][ds] = data_repo_np_ptds.X.shape[1:]
            for ds2 in ds_keys[1:]:
                assert self.input_shape_dict[pt][ds_keys[0]] == self.input_shape_dict[pt][ds2]
            self.input_shape_dict[pt] = list(self.input_shape_dict[pt][ds_keys[0]])
        self.data_generator_obj = MultiSourceDataGenerator
        self.data_generator_additional_args = {"train":{}, "nontrain":{}}
        self.config["training_params"]["config_name"] = self.config["name"]

        # if there is global_config, override some params
        with open(os.path.join(path_definitions.CONFIG_PATH, f"global_config.yaml"), "r") as f:
            self.global_config = yaml.safe_load(f)
        if ("training_params" in self.global_config['dl']):
            if ("best_epoch_strategy" in self.global_config['dl']["training_params"]):
                self.config["training_params"]["best_epoch_strategy"] = \
                    self.global_config["dl"]["training_params"]["best_epoch_strategy"]
            if ("skip_training" in self.global_config['dl']["training_params"]):
                self.config["training_params"]["skip_training"] = \
                    self.global_config["dl"]["training_params"]["skip_training"]

        # Extra pids dict for overlap pid training and test
        with open(os.path.join(path_definitions.DATA_PATH, "additional_user_setup", "overlapping_pids.json"), "r") as f:
            self.overlapping_pids_dict = json.load(f)
        # Extra pids dict for single dataset split
        with open(os.path.join(path_definitions.DATA_PATH, "additional_user_setup", "split_5fold_pids.json"), "r") as f:
            self.split_5fold_pids_dict = json.load(f)


    def prep_data_repo(self, dataset:DatasetDict, flag_train:bool = True) -> DataRepo_tf:
        """ Prepare data repo for deep learning methods """
        super().prep_data_repo(flag_train = flag_train)
        tf.keras.utils.set_random_seed(42)

        self.config["data_loader"]["training_dataset_key"] = dataset.key
        ds_keys = dataset.key.split(":")

        pred_target = dataset.prediction_target
        input_shape = self.input_shape_dict[pred_target]
        ds_key_all = self.data_repo_np_dict[pred_target].keys()
        self.input_shape = input_shape

        if (hasattr(dataset, "eval_task")):
            self.config["training_params"]["eval_task"] = dataset.eval_task
        else:
            self.config["training_params"]["eval_task"] = ""

        # extra processing for overlapping pids
        # only effective for overlapping users evaluation pipeline
        flag_overlap_filter = hasattr(dataset, "flag_overlap_filter")
        if flag_overlap_filter:
            overlap_group, overlap_ds_key_train = dataset.flag_overlap_filter.split(":")
            assert overlap_group in ["test", "train"]
            assert len(ds_keys) == 1

        # extra processing for cross validation split
        # only effective for single dataset evaluation pipeline
        flag_split_filter = hasattr(dataset, "flag_split_filter")
        if flag_split_filter:
            split_group, split_fold = dataset.flag_split_filter.split(":")
            assert split_group in ["test", "train"]

        # extra processing for single daset within user split
        # only effective for single dataset within user evaluation pipeline
        flag_single_within_user_split = hasattr(dataset, "flag_single_within_user_split")
        if flag_single_within_user_split:
            within_split_group, within_split_ratio = dataset.flag_single_within_user_split.split(":")
            assert within_split_group in ["train", "test"]
            if within_split_group == "train":
                within_split_ratio = float(within_split_ratio)
            else:
                within_split_ratio = 1 - float(within_split_ratio)
            assert (0 < within_split_ratio) and (within_split_ratio < 1)
            assert len(ds_keys) == 1

        # at most one flag at one time
        assert np.sum([1 if flag else 0 for flag in [flag_overlap_filter,flag_split_filter,flag_single_within_user_split]]) <= 1

        data_repo_dict = {}
        for ds_key in ds_keys:
            data_repo_dict[ds_key] = self.data_repo_np_dict[pred_target][ds_key]
            if flag_overlap_filter:
                idx = np.where([p in set(self.overlapping_pids_dict[dataset.prediction_target][overlap_ds_key_train][ds_key]) for p in data_repo_dict[ds_key].pids])[0]
                data_repo_dict[ds_key] = data_repo_dict[ds_key][idx]
            if flag_split_filter:
                if split_group == "test":
                    split_pids = np.array(self.split_5fold_pids_dict[dataset.prediction_target][ds_key][str(split_fold)])
                elif split_group == "train":
                    split_pids = [self.split_5fold_pids_dict[dataset.prediction_target][ds_key][str(i)] for i in range(1,6) if i!=int(split_fold)]
                    split_pids = np.concatenate(split_pids)
                idx = np.where([p in set(split_pids) for p in data_repo_dict[ds_key].pids])[0]
                data_repo_dict[ds_key] = data_repo_dict[ds_key][idx]
            if flag_single_within_user_split:
                df_data_idx = pd.DataFrame(self.data_repo_np_dict[pred_target][ds_key].pids, columns=['pid']).groupby("pid").apply(lambda x : np.array(x.index))
                df_data_idx = pd.DataFrame(df_data_idx, columns=["idx"])
                df_data_idx["length"] = df_data_idx["idx"].apply(lambda x : len(x))
                df_data_idx['length_test'] = df_data_idx["length"].apply(lambda x : int(np.ceil(x * (1-within_split_ratio))))
                df_data_idx['idx_train'] = df_data_idx.apply(lambda row : row['idx'][:-row["length_test"]], axis = 1)
                df_data_idx['idx_test'] = df_data_idx.apply(lambda row : row['idx'][-row["length_test"]:], axis = 1)
                if within_split_group == "train":
                    data_idx = np.concatenate(df_data_idx["idx_train"].values)
                elif within_split_group == "test":
                    data_idx = np.concatenate(df_data_idx["idx_test"].values)
                data_repo_dict[ds_key] = data_repo_dict[ds_key][data_idx]
                
        if (flag_train):
            ###############
            # prepare a data generator for feed data into deep models
            ###############

            # get the complete dataset and then split them into train and validation set
            data_generator_whole = self.data_generator_obj(data_repo_dict = data_repo_dict,
                                                is_training=False,
                                                batch_size=1, shuffle=False,
                                                flag_y_vector=self.config["model_params"]["flag_y_vector"],
                                                **self.data_generator_additional_args.get("nontrain", {}),
                                                )
            self.data_generator_whole = data_generator_whole
            # randomly take one sample from each person and use it as validation
            data_repo_dict_train = {}
            data_repo_dict_val = {}
            for ds_key in ds_keys:
                pids_tmp = data_generator_whole.person_list_dict[ds_key]
                data_idx_train = []
                data_idx_val = []
                for pid in pids_tmp:
                    info = data_generator_whole.person_dict[pid]
                    data_idx = list(info["data_idx"])
                    if (len(data_idx) == 1): # person only have one datapoint, val will be the same as train
                        data_idx_val.append(data_idx[0])
                        data_idx_train += data_idx
                    else:
                        random_val_idx = np.random.randint(0, len(data_idx))
                        data_idx_val.append(data_idx[random_val_idx])
                        data_idx_train += data_idx[:random_val_idx] + data_idx[(random_val_idx+1):]
                data_repo_dict_train[ds_key] = data_repo_dict[ds_key][data_idx_train]
                data_repo_dict_val[ds_key] = data_repo_dict[ds_key][data_idx_val]

            # training data generator
            data_generator_train = self.data_generator_obj(data_repo_dict = data_repo_dict_train,
                                                is_training=True,
                                                shuffle=True,
                                                flag_y_vector = self.config["model_params"]["flag_y_vector"],
                                                **self.config["data_loader"], **self.data_generator_additional_args.get("train", {}),
                                                )
            # data generator with the same training data (for validation purpose)
            data_generator_valtrain = self.data_generator_obj(data_repo_dict = data_repo_dict_train,
                                                is_training=False,
                                                batch_size=1, shuffle=False,
                                                flag_y_vector = self.config["model_params"]["flag_y_vector"],
                                                **self.data_generator_additional_args.get("nontrain", {}),
                                                )
            # data generator with the validation data (for validation purpose)
            data_generator_val = self.data_generator_obj(data_repo_dict = data_repo_dict_val,
                                                is_training=False,
                                                batch_size=1, shuffle=False,
                                                flag_y_vector = self.config["model_params"]["flag_y_vector"],
                                                **self.data_generator_additional_args.get("nontrain", {}),
                                                )
            # wrap into the tensorflow API
            dataset_tf_whole = tf.data.Dataset.from_generator(data_generator_whole,
                            output_signature=data_generator_whole.tf_output_signature)
            dataset_tf_train = tf.data.Dataset.from_generator(data_generator_train,
                            output_signature=data_generator_train.tf_output_signature)
            dataset_tf_valtrain = tf.data.Dataset.from_generator(data_generator_valtrain,
                            output_signature=data_generator_train.tf_output_signature)
            dataset_tf_val = tf.data.Dataset.from_generator(data_generator_val,
                            output_signature=data_generator_val.tf_output_signature)
            dataset_tf = {
                "train": dataset_tf_train,
                "valtrain": dataset_tf_valtrain,
                "val": dataset_tf_val,
                "val_whole": dataset_tf_whole
            }

            # If using all but one datasets for training, including the rest for testing.
            # Under special setup (with a bit information leakage), the test set can be used for picking the "optimal" epoch
            if (len(ds_keys) == len(ds_key_all) - 1):
                ds_key_rest = [i for i in ds_key_all if i not in ds_keys][0]
                ds_rest_dict = {ds_key_rest:self.data_repo_np_dict[pred_target][ds_key_rest]}
                if flag_overlap_filter:
                    idx = np.where([p in set(self.overlapping_pids_dict[dataset.prediction_target][overlap_ds_key_train][ds_key_rest]) for p in ds_rest_dict[ds_key_rest].pids])[0]
                    ds_rest_dict[ds_key_rest] = ds_rest_dict[ds_key_rest][idx]
                data_generator_test = self.data_generator_obj(data_repo_dict = ds_rest_dict,
                                                    is_training=False,
                                                    batch_size=1, shuffle=False,
                                                    flag_y_vector = self.config["model_params"]["flag_y_vector"],
                                                    **self.data_generator_additional_args.get("nontrain", {}),
                                                    )
                dataset_tf_test = tf.data.Dataset.from_generator(data_generator_test,
                                output_signature=data_generator_test.tf_output_signature)
                dataset_tf.update({"test": dataset_tf_test})       
        else:
            # If only prepare the testing set, the process is simpler
            data_generator = self.data_generator_obj(data_repo_dict = data_repo_dict,
                                                is_training=False,
                                                batch_size=1, shuffle=False,
                                                flag_y_vector = self.config["model_params"]["flag_y_vector"],
                                                **self.data_generator_additional_args.get("nontrain", {}),
                                                )

            dataset_tf = tf.data.Dataset.from_generator(data_generator,
                            output_signature=data_generator.tf_output_signature)


        if (flag_train):
            # If training, the dataset_tf is a dictionary of data generator
            # The classifier will take care of it properly
            self.data_repo = DataRepo_tf(X = dataset_tf, y = None, pids = None)
        else:
            # If testing, the dataset_tf is just one data generator
            for data_tmp, label_tmp in dataset_tf: break # take one step
            self.data_repo = DataRepo_tf(X = dataset_tf,
                y = np.argmax(data_tmp["input_y"], axis=1) if self.config["model_params"]["flag_y_vector"] else np.array(data_tmp["input_y"]),
                pids = np.array(data_tmp["input_person"]))

        return self.data_repo
    
    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        """ Define a deep-leanrning classifier and add additional necessary parammeters """
        super().prep_model()

        self.config["model_params"].update(
            {"input_shape": self.input_shape,
            "flag_return_embedding":False, "flag_embedding_norm":False,
            "flag_input_dict":True}
        )
        return DepressionDetectionClassifier_DL_erm(config = self.config)
