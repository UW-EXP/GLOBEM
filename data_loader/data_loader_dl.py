import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from data_loader import data_loader_ml
from data_loader.data_loader_ml import DatasetDict, DataRepo, DataRepo_np

from utils import path_definitions


class MultiSourceDataGenerator():
    """ Data Generator for deep model training and evaluation """
    def __init__(self, data_repo_dict: Dict[str, DataRepo], is_training = True,
                generate_by = "across_dataset", 
                batch_size=32, shuffle=True, flag_y_vector=True,
                mixup = "across", mixup_alpha=0.2, **kwargs):
        
        self.X_dict = {k:v.X for k, v in data_repo_dict.items()}
        self.y_dict = {k:v.y for k, v in data_repo_dict.items()}
        self.pids_dict = {k:v.pids for k, v in data_repo_dict.items()}
        self.is_training = is_training
        self.dataset_list = list(self.X_dict.keys())
        self.dataset_dict = {k:idx for idx , k in enumerate(self.dataset_list)}
        
        # Define individual information
        # which will be used for individual data feeding setup
        self.person_list_dict = {k:sorted(list(set(v))) for k, v in self.pids_dict.items()}
        person_dict_tmp = {p:{"k":k,"i":np.where(self.pids_dict[k] == p)[0]} for k,v in self.person_list_dict.items() for p in v}
        self.person_dict = {}
        person_counter = 0
        for pid in itertools.chain.from_iterable(itertools.zip_longest(*list(self.person_list_dict.values()))):
            if (not pid): continue
            self.person_dict[pid] = {"person_idx": person_counter, "dataset_key": person_dict_tmp[pid]["k"],
                "data_idx": person_dict_tmp[pid]["i"], "data_len": len(person_dict_tmp[pid]["i"])}
            person_counter += 1
        self.person_datalen_list = [v["data_len"] for pid ,v in self.person_dict.items()]
        self.person_idx_dict = {k: np.array([self.person_dict[p]["person_idx"] for p in v]) for k, v in self.pids_dict.items()}
        self.person_list = list(self.person_dict.keys())

        self.X_dim = len(self.X_dict[self.dataset_list[0]].shape) - 1
        self.flag_y_vector = flag_y_vector
        self.sample_num_dict = {k:len(self.X_dict[k]) for k in self.X_dict}
        self.sample_num_min = min(self.sample_num_dict.values())

        # Define the generator type
        # within_person: generate data one person' data at one step
        # across_person: generate data for multiple people at one step
        # within_dataset: generate data within one dataset at one step
        # across_dataset: generate data across multiple datasets at one step
        self.generate_by = generate_by
        assert self.generate_by in ["within_person", "across_person", "within_dataset", "across_dataset"]
        self.mixup_alpha = mixup_alpha
        self.shuffle = shuffle
        self.mixup = mixup
        if (self.generate_by == "across_dataset"):
            assert self.mixup in ["across", "within", None]
        else:
            assert self.mixup in ["within", None]

        # Define batch size based on different data generation setup
        self.batch_size_total = batch_size
        if (self.generate_by == "across_dataset"):
            if (self.mixup == "across"):
                self.step_size = self.batch_size_total
                self.step_per_epoch = self.sample_num_min // self.batch_size_total
                # when there is only one dataset, degrade to within mixup
                if (len(self.dataset_list) == 1):
                    self.generate_by = "within_dataset"
                    self.step_size = self.batch_size_total
                    self.step_per_epoch = self.sample_num_min // self.batch_size_total * len(self.dataset_list)
            else:
                self.step_size = batch_size // len(self.X_dict)
                self.step_per_epoch = self.sample_num_min // self.step_size
        elif (self.generate_by == "within_dataset"):
            self.step_size = self.batch_size_total
            self.step_per_epoch = self.sample_num_min // self.batch_size_total * len(self.dataset_list)
        elif (self.generate_by == "within_person"):
            self.step_size = self.batch_size_total
            self.step_per_epoch = len(self.person_dict)
        elif (self.generate_by == "across_person"):
            self.step_size = self.batch_size_total
            self.step_per_epoch = max(self.person_datalen_list, key = self.person_datalen_list.count) // self.batch_size_total
        if (self.step_per_epoch == 0):
            self.step_per_epoch = 1
        
        self.iter_counter = 0

        # define the output shape of the generator
        self.input_shape = list(self.X_dict[self.dataset_list[0]].shape[1:])
        self.tf_output_signature = ({
                        "input_X": tf.TensorSpec(shape=[None] + self.input_shape, dtype = tf.float64),
                        "input_y": tf.TensorSpec(shape=(None, 2) if self.flag_y_vector else (None), dtype = tf.float64),
                        "input_dataset": tf.TensorSpec(shape=(None), dtype = tf.int64),
                        "input_person": tf.TensorSpec(shape=(None), dtype = tf.int64),
                    }, tf.TensorSpec(shape=(None, 2) if self.flag_y_vector else (None), dtype = tf.float64))

    def __call__(self):
        while True:
            indexes_dict = self.__get_exploration_order()

            # if val/test, just return one step with all data
            if (not self.is_training):
                X = np.concatenate(list(self.X_dict.values()))
                y = np.concatenate(list(self.y_dict.values()))
                dsidx = np.zeros(len(y))
                personidx = np.zeros(len(y))

                dsidx = np.concatenate([self.dataset_dict[ds] * np.ones(len(self.y_dict[ds])) for ds in self.dataset_list])
                personidx = np.array([self.person_dict[p]["person_idx"] for ds in self.dataset_list for p in self.pids_dict[ds]])

                if (not self.flag_y_vector):
                    y = np.argmax(y, axis = 1)
                yield {"input_X":X, "input_y":y, "input_dataset": dsidx, "input_person": personidx}, y
                break
            else:
                # if train, return data based on different generator types
                if (self.generate_by == "across_dataset"):
                    if (self.mixup == "across"):
                        for i in range(self.step_per_epoch):
                            batch_ids_dict = {}
                            for k, indexes in indexes_dict.items():
                                batch_ids_dict[k] = indexes[i * self.step_size:(i + 1) * self.step_size]
                            X, y = self.__data_generation_between(batch_ids_dict)
                            if (not self.flag_y_vector):
                                y = np.argmax(y, axis = 1)
                            # due to the mixup, it's hard to maintain the ds and person idx
                            dsidx = np.zeros(len(y))
                            personidx = np.zeros(len(y))
                            yield {"input_X":X, "input_y":y, "input_dataset": dsidx, "input_person": personidx}, y
                    else:
                        for i in range(self.step_per_epoch):
                            X_dict = {}
                            y_dict = {}
                            dataset_dict = {}
                            person_dict = {}
                            for k, indexes in indexes_dict.items():
                                batch_ids = indexes[i * self.step_size:(i + 1) * self.step_size]
                                X, y = self.__data_generation_within(k, batch_ids)
                                X_dict[k] = X
                                y_dict[k] = y
                                dataset_dict[k] = self.dataset_dict[k] * np.ones(len(y))
                                person_dict[k] = self.person_idx_dict[k][batch_ids]
                            X = np.concatenate(list(X_dict.values()))
                            y = np.concatenate(list(y_dict.values()))
                            dsidx = np.concatenate(list(dataset_dict.values()))
                            personidx = np.concatenate(list(person_dict.values()))
                            if (not self.flag_y_vector):
                                y = np.argmax(y, axis = 1)
                            if (self.mixup is not None): # people are mixed
                                personidx = np.zeros(len(y))
                            yield {"input_X":X, "input_y":y, "input_dataset": dsidx, "input_person":personidx}, y
                elif (self.generate_by == "within_dataset"):
                    for i in range(self.step_per_epoch):
                        dataset_idx = i % len(self.dataset_list)
                        dataset_key = self.dataset_list[dataset_idx]
                        j = i // len(self.dataset_list)
                        batch_ids = indexes_dict[dataset_key][j * self.step_size:(j + 1) * self.step_size]
                        X, y = self.__data_generation_within(dataset_key, batch_ids)
                        dsidx = dataset_idx * np.ones(len(y))
                        personidx = self.person_idx_dict[dataset_key][batch_ids]

                        if (not self.flag_y_vector):
                            y = np.argmax(y, axis = 1)
                        if (self.mixup is not None): # people are mixed
                            personidx = np.zeros(len(y))
                        yield {"input_X":X, "input_y":y, "input_dataset": dsidx, "input_person": personidx}, y
                elif (self.generate_by == "within_person"):
                    for i in range(self.step_per_epoch):
                        persons = self.person_list[i: (i+1)]
                        X_dict = {}
                        y_dict = {}
                        dataset_dict = {}
                        person_dict = {}
                        for person in persons:
                            info = self.person_dict[person]
                            k = info["dataset_key"]
                            batch_ids = info["data_idx"]
                            if (self.step_size is not None and self.step_size < len(batch_ids)):
                                batch_ids = np.random.choice(batch_ids, size=self.step_size, replace = False)                                
                            X, y = self.__data_generation_within(k, batch_ids)
                            X_dict[person] = X
                            y_dict[person] = y
                            dataset_dict[person] = self.dataset_dict[k] * np.ones(len(y))
                            person_dict[person] = info["person_idx"] * np.ones(len(y))
                        X = np.concatenate(list(X_dict.values()))
                        y = np.concatenate(list(y_dict.values()))
                        dsidx = np.concatenate(list(dataset_dict.values()))
                        personidx = np.concatenate(list(person_dict.values()))

                        if (not self.flag_y_vector):
                            y = np.argmax(y, axis = 1)
                        yield {"input_X":X, "input_y":y, "input_dataset": dsidx, "input_person": personidx}, y
                elif (self.generate_by == "across_person"):
                    for i in range(self.step_per_epoch):
                        X_dict = {}
                        y_dict = {}
                        dataset_dict = {}
                        person_dict = {}
                        for person in self.person_list:
                            info = self.person_dict[person]
                            k = info["dataset_key"]
                            ids_raw = info["data_idx"]
                            ids_raw_len = len(ids_raw)
                            if (self.step_size <= ids_raw_len):
                                start = (i * self.step_size) % ids_raw_len
                                end = ((i+1)*self.step_size) % ids_raw_len
                                if (start < end):
                                    batch_ids = ids_raw[start : end]
                                else:
                                    batch_ids = np.concatenate([ids_raw[start :], ids_raw[: end]])
                            else:
                                batch_ids_first = np.concatenate([ids_raw for _ in range(self.step_size // ids_raw_len)])
                                batch_ids_second = ids_raw[: (self.step_size % ids_raw_len)]
                                batch_ids = np.concatenate([batch_ids_first, batch_ids_second])
                            X, y = self.__data_generation_within(k, batch_ids)
                            X_dict[person] = X
                            y_dict[person] = y
                            dataset_dict[person] = self.dataset_dict[k] * np.ones(len(y))
                            person_dict[person] = info["person_idx"] * np.ones(len(y))
                        X = np.concatenate(list(X_dict.values()))
                        y = np.concatenate(list(y_dict.values()))
                        dsidx = np.concatenate(list(dataset_dict.values()))
                        personidx = np.concatenate(list(person_dict.values()))
                        if (not self.flag_y_vector):
                            y = np.argmax(y, axis = 1)
                        yield {"input_X":X, "input_y":y, "input_dataset": dsidx, "input_person": personidx}, y                        

            self.iter_counter += 1
                
    def __get_exploration_order(self):
        """ Shuffle data when necessary """
        indexes_dict = {k: np.arange(v) for k, v in self.sample_num_dict.items()}
        # indexes = np.arange(self.sample_num)
        if self.shuffle and self.is_training:
            for k in indexes_dict:
                np.random.shuffle(indexes_dict[k])
            np.random.shuffle(self.person_list)
        return indexes_dict

    def __data_generation_within(self, dataset_key, batch_ids):
        """ Generate mixup data within datasets """
        
        X1 = self.X_dict[dataset_key][batch_ids]
        y1 = self.y_dict[dataset_key][batch_ids]
        
        if self.mixup == "within":
            X2 = self.X_dict[dataset_key][np.random.permutation(batch_ids)]
            y2 = self.y_dict[dataset_key][np.random.permutation(batch_ids)]
            X, y = self.__mixup(X1, y1, [], X2, y2, [])
        else:
            X = X1
            y = y1

        return X, y
    
    def __data_generation_between(self, batch_ids_dict):
        """ Generate mixup data across datasets """
        X1, y1, dataset1 = [], [], []
        X2, y2, dataset2 = [], [], []
        length = min([len(batch_ids_dict[k]) for k in batch_ids_dict])
        for _ in range(length):
            dataset_key1, dataset_key2 = np.random.choice(self.dataset_list, 2, replace=False)
            idx1, idx2 = np.random.randint(low=0, high=length, size=2)
            X1.append(self.X_dict[dataset_key1][batch_ids_dict[dataset_key1][idx1]])
            y1.append(self.y_dict[dataset_key1][batch_ids_dict[dataset_key1][idx1]])
            dataset1.append(dataset_key1)
            X2.append(self.X_dict[dataset_key2][batch_ids_dict[dataset_key2][idx2]])
            y2.append(self.y_dict[dataset_key2][batch_ids_dict[dataset_key2][idx2]])
            dataset2.append(dataset_key2)
        X, y = self.__mixup(X1, y1, dataset1, X2, y2, dataset2)
        return X, y
            
    
    def __mixup(self, X1, y1, dataset1, X2, y2, dataset2):
        """ Mixuping data of two sides """
        l = np.random.beta(self.mixup_alpha, self.mixup_alpha, len(X1))
        if (self.X_dim == 3):
            X_l = l.reshape(len(X1), 1, 1, 1)
        elif (self.X_dim == 2):
            X_l = l.reshape(len(X1), 1, 1)
        elif (self.X_dim == 1):
            X_l = l.reshape(len(X1), 1)
        else:
            print("X_dim seems very large")
        y_l = l.reshape(len(X1), 1)
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)

        return X, y

def normalize_along_axis(data: np.ndarray, axis:int = -2, method:str = "robust") -> np.ndarray:
    """Normalize the data along a given axis

    Args:
        data (np.ndarray): dataframe to be normalized
        axis (int, optional): dimension to be normalized along. Defaults to -2.
        method (str, optional): current support "standard" (minus mean and std) 
            or "robust" (minus median and divided 5-95 quantile range). Defaults to "robust".

    Returns:
        np.ndarray: normalized dataframe
    """
    if (method == "standard"):
        return (data - np.mean(data, axis = axis, keepdims=True)) / (np.std(data, axis = axis, keepdims=True) + 1e-9)
    elif (method == "robust"):
        q_small, q_center, q_large = np.nanpercentile(data, q = [5,50,95], axis = axis, keepdims=True)
        r = q_large - q_small + 1e-9
        data_scale = (data - q_center) / r
        return np.clip(data_scale, a_min = -2, a_max = 2)

def data_loader_np(ds_keys_dict: dict, flag_normalize:bool = True, flag_more_feat_types:bool = False, verbose:bool = True) -> Dict[str, Dict[str, DataRepo_np]]:
    """Prep a dictionary of DataRepo_np for deep learning purpose

    Args:
        ds_keys_dict (dictionary): a dictionary of <pred_target, ds_keys> pairs
        flag_normalize (bool, optional): whether to use normalized features. Defaults to True.
        flag_more_feat_types (bool, optional): whether load all sensor types.
            Should be False for maximum compatibility. Defaults to False.
        verbose (bool, optional): Whether to display the progress bar and intermediate reuslts. Defaults to True

    Raises:
        ValueError: Incompatible input shape

    Returns:
        Dict[str, Dict[str, DataRepo_np]]: a dictionary of dictionary of DataRepo_np,
            with the first level as prediction target, and the second level as ds_key
    """
    
    if (not flag_normalize):
        data_repo_np_dict = {}
        for pred_target, ds_keys in tqdm(ds_keys_dict.items(), position=0, desc= "prediction targets", disable=not verbose):
            for ds_key in tqdm(ds_keys, position=1, desc= "dataset keys", leave=False, disable= not verbose):
                institution, phase = ds_key.split("_")
                phase = int(phase)
                if pred_target not in data_repo_np_dict:
                    data_repo_np_dict[pred_target] = {}
                if flag_more_feat_types:
                    dataset_file_np_path = os.path.join(path_definitions.DATA_PATH, "np_max_feature_types", f"{pred_target}--{ds_key}--np.pkl")
                else:
                    dataset_file_np_path = os.path.join(path_definitions.DATA_PATH, "np", f"{pred_target}--{ds_key}--np.pkl")
                if (os.path.exists(dataset_file_np_path)):
                    if (verbose):
                        tqdm.write(pred_target + " " + ds_key + " read np " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    with open(dataset_file_np_path, "rb") as f:
                        data_repo_np_dict[pred_target][ds_key] = pickle.load(f)
                else:
                    dataset = data_loader_ml.data_loader_single(pred_target, institution, phase,
                        flag_more_feat_types=flag_more_feat_types)
                    feat_prep = dl_feat_preparation(flag_use_features="both",
                                                    flag_feature_selection=None,
                                                    flag_more_feat_types=flag_more_feat_types,
                                                    verbose=1 if verbose else 0)
                    if (verbose):
                        tqdm.write(pred_target + " " + ds_key + " compute np " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    data_repo_np = DataRepo_np(feat_prep.prep_data_repo(dataset),
                        cols = feat_prep.feature_list)
                    Path(os.path.split(dataset_file_np_path)[0]).mkdir(parents=True, exist_ok=True)
                    with open(dataset_file_np_path, "wb") as f:
                        pickle.dump(data_repo_np, f)
                    data_repo_np_dict[pred_target][ds_key] = deepcopy(data_repo_np)
        return data_repo_np_dict
    else:
        data_repo_np_norm_dict = {}
        for pred_target, ds_keys in tqdm(ds_keys_dict.items(), position=0, desc= "prediction targets", disable=not verbose):
            for ds_key in tqdm(ds_keys, position=1, desc= "dataset keys", leave=False, disable=not verbose):
                institution, phase = ds_key.split("_")
                phase = int(phase)
                if pred_target not in data_repo_np_norm_dict:
                    data_repo_np_norm_dict[pred_target] = {}
                if flag_more_feat_types:
                    dataset_file_np_path = os.path.join(path_definitions.DATA_PATH, "np_norm_max_feature_types", f"{pred_target}--{ds_key}--np.pkl")
                    dataset_file_np_path_nonorm = os.path.join(path_definitions.DATA_PATH, "np_max_feature_types", f"{pred_target}--{ds_key}--np.pkl")
                else:
                    dataset_file_np_path = os.path.join(path_definitions.DATA_PATH, "np_norm", f"{pred_target}--{ds_key}--np_norm.pkl")
                    dataset_file_np_path_nonorm = os.path.join(path_definitions.DATA_PATH, "np", f"{pred_target}--{ds_key}--np.pkl")
                if (os.path.exists(dataset_file_np_path)):
                    if (verbose):
                        tqdm.write(pred_target + " " + ds_key + " read np norm " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    with open(dataset_file_np_path, "rb") as f:
                        data_repo_np_norm_dict[pred_target][ds_key] = pickle.load(f)
                else:
                    dataset = data_loader_ml.data_loader_single(pred_target, institution, phase,
                        flag_more_feat_types=flag_more_feat_types)
                    feat_prep = dl_feat_preparation(flag_use_features="both",
                                                    flag_feature_selection=None,
                                                    flag_more_feat_types=flag_more_feat_types,
                                                    verbose=1 if verbose else 0)
                    if (verbose):
                        tqdm.write(pred_target + " " + ds_key + " compute np norm " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

                    data_repo_np = DataRepo_np(feat_prep.prep_data_repo(dataset),
                        cols = feat_prep.feature_list)
                    Path(os.path.split(dataset_file_np_path_nonorm)[0]).mkdir(parents=True, exist_ok=True)
                    with open(dataset_file_np_path_nonorm, "wb") as f:
                        pickle.dump(data_repo_np, f)
                    # Ignore the norm features as they are already normalized on each individual's behavior
                    feature_idx_tobenormed = [idx for idx,f in enumerate(feat_prep.feature_list) if "_norm:" not in f]
                    data_repo_np_norm_dict[pred_target][ds_key] = deepcopy(data_repo_np)
                    X_shape = data_repo_np_norm_dict[pred_target][ds_key].X.shape
                    if (len(X_shape) == 3):
                        data_repo_np_norm_dict[pred_target][ds_key].X[:,:,feature_idx_tobenormed] = \
                            normalize_along_axis(data_repo_np_norm_dict[pred_target][ds_key].X[:,:,feature_idx_tobenormed], axis = -2, method = "robust")
                    elif (len(X_shape) == 2):
                        data_repo_np_norm_dict[pred_target][ds_key].X[:,feature_idx_tobenormed] = \
                            normalize_along_axis(data_repo_np_norm_dict[pred_target][ds_key].X[:,feature_idx_tobenormed], axis = -2, method = "robust")
                    else:
                        raise ValueError(f"X's shape is {X_shape}")
                    Path(os.path.split(dataset_file_np_path)[0]).mkdir(parents=True, exist_ok=True)
                    with open(dataset_file_np_path, "wb") as f:
                        pickle.dump(data_repo_np_norm_dict[pred_target][ds_key], f)
        return data_repo_np_norm_dict

def prep_repo_np_dict_feature_prep(data_repo_np_dict:Dict[str, Dict[str, DataRepo_np]],
    ndim:int = 2, selected_feature_idx:List[int] = None) -> Dict[str, Dict[str, DataRepo_np]]:
    """ Take features and process dimensions when necessary """
    for pred_target in data_repo_np_dict:
        ds_keys = list(data_repo_np_dict[pred_target].keys())
        new_feature_idx = list(np.arange(data_repo_np_dict[pred_target][ds_keys[0]].X.shape[-1]))
        if (selected_feature_idx):
            new_feature_idx = list(selected_feature_idx)

        for ds_key in ds_keys:
            data_repo_np_dict[pred_target][ds_key].X = data_repo_np_dict[pred_target][ds_key].X[:,:,new_feature_idx]

        if (ndim == 1): # aggregate across days 
            for k, v in data_repo_np_dict[pred_target].items():
                data_repo_np_dict[pred_target][k].X = np.concatenate([np.mean(v.X, axis = 1), np.std(v.X, axis = 1)], axis=-1)
        elif (ndim == 3): # simply define the num of channels to be 1
            for k, v in data_repo_np_dict[pred_target].items():
                data_repo_np_dict[pred_target][k].X = np.expand_dims(v.X, axis = -1)
        else: # do nothing
            pass

    return data_repo_np_dict

def data_loader_dl_placeholder(pred_targets: List[str], ds_keys_target: List[str], verbose:bool = True):
    """ Load the data placeholder when doing a training.
        This can accelerate the process as dl model will load np instead """
    datadict_filepath = os.path.join(path_definitions.DATA_PATH, "dataset_dict_dl_placeholder.pkl")

    def generate_dl_placeholder():
        ds_keys = global_config["all"]["ds_keys"]

        dataset_dict = data_loader_ml.data_loader({pt: ds_keys for pt in global_config["all"]["prediction_tasks"]}, verbose=verbose)
        for pt, dsd_ds in dataset_dict.items():
            for ds, dsd in dsd_ds.items():
                dataset_dict[pt][ds].datapoints = dataset_dict[pt][ds].datapoints.iloc[:2]
        with open(datadict_filepath, "wb") as f:
            pickle.dump(dataset_dict, f)
        return dataset_dict

    if (os.path.exists(datadict_filepath)):
        try:
            with open(datadict_filepath, "rb") as f:
                dataset_dict = pickle.load(f)
            assert set(pred_targets).issubset(set(dataset_dict.keys()))
            for pred_target in pred_targets:
                assert set(ds_keys_target).issubset(set(dataset_dict[pred_target].keys()))
        except:
            dataset_dict = generate_dl_placeholder()
    else:
        dataset_dict = generate_dl_placeholder()
    return dataset_dict


class dl_feat_preparation():
    """A class to help feature perparation for deep learning models """

    def __init__(self, config_name = "dl_feat_prep", flag_more_feat_types = False, verbose = 0, **kwargs):
        super().__init__()

        with open(os.path.join(path_definitions.CONFIG_PATH, f"{config_name}.yaml"), "r") as f:
            self.config = yaml.safe_load(f)

        all_feats = []
        if (flag_more_feat_types):
            feature_type_list = ['f_loc', 'f_screen', 'f_slp', 'f_steps', "f_blue", "f_call"]
        else:
            feature_type_list = ['f_loc', 'f_screen', 'f_slp', 'f_steps']
        for epoch in epochs_5:
            all_feats += [f for ft in feature_type_list for f in fc_repo.feature_columns_selected_epoches_types[epoch][ft]]

        self.feature_list_nonorm = deepcopy(all_feats)
        self.feature_list_norm = []
        for f in all_feats:
            ft, fn, seg = f.split(":")
            new_f = f"{ft}:{fn}_norm:{seg}"
            self.feature_list_norm.append(new_f)
        self.feature_list = self.feature_list_nonorm + self.feature_list_norm

        if (flag_more_feat_types):
            self.selected_feature_list = self.config["feature_definition"]["feature_list_more_feat_types"]
        else:
            self.selected_feature_list = self.config["feature_definition"]["feature_list"]
        self.selected_feature_idx = [self.feature_list.index(f) for f in self.selected_feature_list]

        self.NAFILL = 0
        self.verbose = verbose
        
    def prep_data_repo_aggregate(self, dataset:DatasetDict, flag_train:bool = True, calc_method="last") -> DataRepo:
        """Basic feature calculation to obtain either calculate median or get the last day's feature value"""

        assert calc_method in ["last", "stats"]

        df_datapoints = deepcopy(dataset.datapoints)
        
        if (calc_method == "last"):
            def get_last(df):
                return pd.Series(data = df[self.feature_list].iloc[-1].values, index = self.feature_list).T
            X_tmp = df_datapoints["X_raw"].apply(lambda x : get_last(x))
        else:
            @ray.remote
            def get_stats(df):
                median_tmp = pd.Series(data = df[self.feature_list].iloc[-14:].median().values, index = [f + "#median" for f in self.feature_list]).T
                return pd.concat([median_tmp])
            
            X_tmp = ray.get([get_stats.remote(df) for df in df_datapoints["X_raw"]])
        X_tmp = pd.DataFrame(X_tmp)
        X_tmp.index = df_datapoints.index

        # filter
        shape1 = X_tmp.shape
        X_tmp = X_tmp[X_tmp.isna().sum(axis = 1) < X_tmp.shape[1] / 2]  # filter very empty days
        shape2 = X_tmp.shape
        del_rows = shape1[0] - shape2[0]

        X = deepcopy(X_tmp)

        if (self.verbose > 0):
            print(f"filter {del_rows} rows")
            print(f"NA rate: {100* X.isna().sum().sum() / X.shape[0] / X.shape[1]}%" )
        X = X.fillna(X.median())
        X = X.fillna(0) # for those completely empty features (e.g., one dataset does not have the feature)

        y = df_datapoints["y_raw"].loc[X.index]
        pids = df_datapoints["pid"].loc[X.index]
        
        self.data_repo = DataRepo(X=X, y=y, pids=pids)
        return self.data_repo

    def prep_data_repo(self, dataset:DatasetDict, flag_train:bool = True) -> DataRepo:
        """Basic feature calculation to obtain median"""
        
        df_datapoints = deepcopy(dataset.datapoints)

        df_datapoints_X = df_datapoints["X_raw"].apply(lambda df : df[self.feature_list].iloc[-28:])

        @globalize
        def impute(df):
            return df.fillna(df.median(axis = 0),axis=0).fillna(self.NAFILL).values

        with Pool(NJOB) as pool:
            results = list(tqdm(pool.imap(impute, df_datapoints_X.values),
                total = len(df_datapoints_X), position = 2, leave=False, desc = "Feature processing", disable=int(self.verbose)==0))

        df_results = [pd.DataFrame(r, index= df_datapoints_X.iloc[0].index, columns= df_datapoints_X.iloc[0].columns) for r in results]

        X = pd.Series(df_results, index=df_datapoints_X.index)

        y = df_datapoints["y_raw"].loc[X.index]
        pids = df_datapoints["pid"].loc[X.index]
        
        self.data_repo = DataRepo(X=X, y=y, pids=pids)
        return self.data_repo