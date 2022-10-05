import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from utils import path_definitions
from data import data_factory

class DatasetDict():
    """A data structor of saving a dataset, including the dataset key, prediction target, 
        as well as datapoints (as a pandas.DataFrame) """
    def __init__(self, key: str, prediction_target: str, datapoints: pd.DataFrame):
        self.key = deepcopy(key)
        self.prediction_target = deepcopy(prediction_target)
        assert type(datapoints) == pd.DataFrame
        assert set(["pid", "date","X_raw","y_raw","device_type"]).issubset(set(datapoints.columns))
        self.datapoints = deepcopy(datapoints)

class DataRepo():
    """A data structor focused on saving data input matrix X, label y, and participant ID (pid)"""
    def __init__(self, X:pd.DataFrame, y:pd.Series, pids:pd.Series):
        self.X = deepcopy(X)
        self.y = deepcopy(y)
        self.pids = deepcopy(pids)

    def __getitem__(self, key):
        return DataRepo(self.X[key], self.y[key], self.pids[key])

class DataRepo_tf(DataRepo):
    """A variant of DataRepo. It has the same structure of DataRepo.
        It acts as a special data type for tensorflow dataset."""
    def __init__(self, X:tf.data.Dataset or Dict[str, tf.data.Dataset], y: List[bool] or np.ndarray, pids: List[bool] or np.ndarray):
        super().__init__(None, None, None)
        self.X = X
        self.y = y
        self.pids = pids

class DataRepo_np(DataRepo):
    """A variant of DataRepo. It only saves input X as np.ndarray.
        So it has the additional structure of saving columns information.
        The main difference between DataRepo and DataRepo_np is that
        the latter is mainly used for deep learning methods to simply processing.
        After processing, a DataRepo_np will be converted to DataRepo_tf, which was then used for model training"""
    def __init__(self, data_repo: DataRepo = None, cols: List[str] = None,
            X:np.ndarray=None, y:np.ndarray=None, pids:np.ndarray=None):
        super().__init__(None, None, None)
        if (data_repo is not None):
            self.X = np.array([i for i in data_repo.X.values])
            self.y = np.array([[0,1] if y else [1,0] for y in data_repo.y.values])
            self.pids = deepcopy(data_repo.pids.values)
            if (cols is not None):
                assert len(cols) == self.X.shape[-1]
            self.X_cols = deepcopy(cols)
        else:
            self.X = X
            self.y = y
            self.pids = pids
            self.X_cols = cols

    def __getitem__(self, key):
        return DataRepo_np(X = self.X[key], y = self.y[key], pids = self.pids[key], cols = self.X_cols)

def data_loader_read_label_file(institution:str, phase:int, prediction_target:str) -> Union[pd.DataFrame,str]:
    """Load a single label file

    Args:
        institution (str): insitution code
        phase (int): number of study phase
        prediction_target (str): prediction task, current support "dep_endterm" and "dep_weekly"
    
    Raises:
        ValueError: an unsupported prediction target

    Returns:
        pd.DataFrame: dataframe of labels
        str: prediction target col name
    """
    if (prediction_target == "dep_weekly"):
        prediction_target_col = "dep"
        df_label = pd.read_csv(data_factory.survey_folder[institution][phase] + "dep_weekly.csv")
    elif (prediction_target == "dep_endterm"):
        prediction_target_col = "dep"
        df_label = pd.read_csv(data_factory.survey_folder[institution][phase] + "dep_endterm.csv")
    else:
        df_label_ema = pd.read_csv(data_factory.survey_folder[institution][phase] + "ema.csv")
        df_label_pre = pd.read_csv(data_factory.survey_folder[institution][phase] + "pre.csv")
        df_label_post = pd.read_csv(data_factory.survey_folder[institution][phase] + "post.csv")
        if prediction_target not in data_factory.threshold_book:
            raise ValueError(f"'{prediction_target}' is not defined in threshold book.")
        threhold_as_true = data_factory.threshold_book[prediction_target]["threshold_as_true"]
        threhold_as_false = data_factory.threshold_book[prediction_target]["threshold_as_false"]
        if (threhold_as_true > threhold_as_false):
            flag_larger_is_true = True
        elif (threhold_as_true < threhold_as_false):
            flag_larger_is_true = False
        else:
            raise ValueError(f"Please specifiy {prediction_target}'s two-side thresholds separately (inclusive). They cannot be the same.")

        # Add new keys into extra pids dict to handle new predict target
        with open(os.path.join(path_definitions.DATA_PATH, "additional_user_setup", "overlapping_pids.json"), "r") as f:
            overlapping_pids_dict = json.load(f)
        with open(os.path.join(path_definitions.DATA_PATH, "additional_user_setup", "split_5fold_pids.json"), "r") as f:
            split_5fold_pids_dict = json.load(f)            

        prediction_target_col = prediction_target + '_label'
        if (prediction_target in df_label_ema.columns):
            df_label = deepcopy(df_label_ema)
            closer_dep_task = "dep_weekly"
        elif (prediction_target in df_label_pre.columns):
            df_label = deepcopy(df_label_pre)
            closer_dep_task = "dep_endterm"
        elif (prediction_target in df_label_post.columns):
            df_label = deepcopy(df_label_post)
            closer_dep_task = "dep_endterm"
        else:
            raise ValueError(f"'{prediction_target}' not in the survey file.")
        df_label = deepcopy(df_label[~df_label[prediction_target].isna()])
        df_label[prediction_target_col] = df_label[prediction_target].apply(
            lambda x : float(x) >= threhold_as_true if flag_larger_is_true else float(x) <= threhold_as_true)
        # simply copy the split from the basic task. Can be improved
        if (prediction_target not in overlapping_pids_dict):
            overlapping_pids_dict[prediction_target] = deepcopy(overlapping_pids_dict[closer_dep_task])
        if (prediction_target not in split_5fold_pids_dict):
            split_5fold_pids_dict[prediction_target] = deepcopy(split_5fold_pids_dict[closer_dep_task])
        with open(os.path.join(path_definitions.DATA_PATH, "additional_user_setup", "overlapping_pids.json"), "w") as f:
            json.dump(overlapping_pids_dict, f)
        with open(os.path.join(path_definitions.DATA_PATH, "additional_user_setup", "split_5fold_pids.json"), "w") as f:
            json.dump(split_5fold_pids_dict, f)          
    
    df_label[prediction_target_col+"_raw"] = df_label[prediction_target_col]
    df_label["pid"] = df_label["pid"].apply(lambda x : f"{x}#{institution}_{phase}")
    df_label = df_label.drop_duplicates(["pid", "date"], keep = "last")
    df_label["date"] = pd.to_datetime(df_label["date"])
    return df_label, prediction_target_col

def data_loader_single_dataset_label_based(institution:str, phase:int,
    prediction_target:str, flag_more_feat_types:bool = False) -> pd.DataFrame:
    """Load a single dataset for DataRepo of a given institution and phase

    Args:
        institution (str): insitution code
        phase (int): number of study phase
        prediction_target (str): prediction task, current support "dep_endterm" and "dep_weekly"
        flag_more_feat_types (bool, optional): whether load all sensor types. Should be False for maximum compatibility. Defaults to False.

    Raises:
        ValueError: an unsupported prediction target

    Returns:
        pd.DataFrame: dataframe of data points that used as DataRepo.datapoints
    """
    df_full_rawdata = pd.read_csv(data_factory.feature_folder[institution][phase] + "rapids.csv", low_memory=False)
    df_full_rawdata["date"] = pd.to_datetime(df_full_rawdata["date"])
    df_full_rawdata["pid"] = df_full_rawdata["pid"].apply(lambda x : f"{x}#{institution}_{phase}")
    
    df_participant_file = pd.read_csv(data_factory.participants_info_folder[institution][phase] + "platform.csv", low_memory=False)
    df_participant_file["pid"] = df_participant_file["pid"].apply(lambda x : f"{x}#{institution}_{phase}")
    df_participant_file = df_participant_file.set_index("pid")

    df_label, prediction_target_col = data_loader_read_label_file(institution, phase, prediction_target)

    datapoints = []

    if not flag_more_feat_types: # maximum compatibility of multiple datasets across insitutes
        fts = ['f_loc', 'f_screen', 'f_slp', 'f_steps']
    else:
        fts = ['f_loc', 'f_screen', 'f_slp', 'f_steps', "f_blue", "f_call"]
    retained_features = ["pid", "date"]
    for col in df_full_rawdata.columns:
        for ft in fts:
            if (col.startswith(ft)):
                retained_features.append(col)
                break
    
    for idx, row in df_label.iterrows():
        pid = row["pid"]
        date_end = row["date"]
        date_start = date_end + timedelta(days = -27) # past 4 weeks

        df_data_window = df_full_rawdata[df_full_rawdata["pid"] == pid]
        df_data_window = df_data_window[(date_start <= df_data_window["date"]) & (df_data_window["date"] <= date_end)]
        if (df_data_window.shape[0] == 0):
            continue
        df_data_windowplaceholder = pd.DataFrame(pd.date_range(date_start, date_end), columns=["date"])
        df_data_windowplaceholder["pid"] = pid
        df_data_window = df_data_windowplaceholder.merge(df_data_window, left_on=["pid","date"], right_on=["pid","date"], how="left")
        df_data_window = deepcopy(df_data_window)

        datapoint = {"pid":pid, "date": date_end,
                     "X_raw": df_data_window[retained_features], "y_raw": row[prediction_target_col], "y_allraw": row,
                     "device_type": df_participant_file.loc[pid]["platform"].split(";")[0] }
        datapoints.append(datapoint)
    df_datapoints = pd.DataFrame(datapoints)

    if (prediction_target == "dep_weekly"):
        pids_few_response = df_datapoints.groupby("pid").count()
        pids_few_response = list(pids_few_response[pids_few_response["date"]<2].index)
        df_datapoints = df_datapoints[~df_datapoints["pid"].isin(pids_few_response)]
    
    return df_datapoints

def data_loader_single_dataset_raw(institution:str, phase:int, prediction_target:str) -> pd.DataFrame:
    """Load a single raw data of a given institution and phase

    Args:
        institution (str): insitution code
        phase (int): number of study phase
        prediction_target (str): prediction task, current support "dep_endterm" and "dep_weekly"

    Raises:
        ValueError: an unsupported prediction target

    Returns:
        pd.DataFrame: dataframe of all raw data, with per person per day as a row
    """
    df_full_rawdata = pd.read_csv(data_factory.feature_folder[institution][phase] + "rapids.csv", low_memory=False)
    df_full_rawdata["date"] = pd.to_datetime(df_full_rawdata["date"])
    df_full_rawdata["pid"] = df_full_rawdata["pid"].apply(lambda x : f"{x}#{institution}_{phase}")

    df_participant_file = pd.read_csv(data_factory.participants_info_folder[institution][phase] + "platform.csv", low_memory=False)
    df_participant_file["pid"] = df_participant_file["pid"].apply(lambda x : f"{x}#{institution}_{phase}")
    df_participant_file = df_participant_file.set_index("pid")

    df_label, prediction_target_col = data_loader_read_label_file(institution, phase, prediction_target)

    retained_features = ["pid", "date"]
    retained_features += [c for c in df_full_rawdata.columns if c not in ["pid", "date"]]

    df_full_rawdata_ = df_full_rawdata.merge(df_label, left_on=["pid","date"], right_on=["pid","date"], how="left")
    df_full_rawdata_ = df_full_rawdata_[[col for col in df_full_rawdata_.columns if col in [prediction_target_col] + retained_features]]
    df_participant_file["platform_split"] = df_participant_file["platform"].apply(lambda x: x.split(";")[0])
    df_full_rawdata_["device_type"] = df_full_rawdata_["pid"].apply(lambda x : df_participant_file.loc[x]["platform_split"])
    df_datapoints = df_full_rawdata_
    
    if (prediction_target == "dep_weekly"):
        pids_few_response = df_datapoints.groupby("pid").count()
        pids_few_response = list(pids_few_response[pids_few_response["date"]<2].index)
        df_datapoints = df_datapoints[~df_datapoints["pid"].isin(pids_few_response)]
    
    return df_datapoints


def data_loader_single(prediction_target:str, institution:str, phase:int, flag_more_feat_types:bool = False) -> DatasetDict:
    """Helper function to load a single DatasetDict of a given institution and phase.
    If the data is already saved as pkl file, load the pkl file directly to accelerate the process.

    Args:
        prediction_target (str): prediction task, current support "dep_endterm" and "dep_weekly"
        institution (str): insitution code
        phase (int): number of study phase
        flag_more_feat_types (bool, optional): whether load all sensor types. Should be False for maximum compatibility. Defaults to False.

    Returns:
        DatasetDict: data structure of a dataset
    """
    ds_key = f"{institution}_{phase}"
    if not flag_more_feat_types:
        dataset_file_path = os.path.join(path_definitions.DATA_PATH, "datarepo", f"{prediction_target}--{ds_key}.pkl")
    else:
        dataset_file_path = os.path.join(path_definitions.DATA_PATH, "datarepo_max_feature_types", f"{prediction_target}--{ds_key}.pkl")

    if (os.path.exists(dataset_file_path)):
        with open(dataset_file_path, "rb") as f:
            dataset = pickle.load(f)
    else:
        datapoints = data_loader_single_dataset_label_based(institution, phase, prediction_target, flag_more_feat_types)
        dataset = DatasetDict(key = ds_key, prediction_target=prediction_target, datapoints=datapoints)
        Path(os.path.split(dataset_file_path)[0]).mkdir(parents=True, exist_ok=True)
        with open(dataset_file_path, "wb") as f:
            pickle.dump(dataset, f)
    return dataset


def data_loader(ds_keys_dict: Dict[str, List[str]], flag_more_feat_types:bool = False, verbose:bool = True) -> Dict[str, Dict[str, DatasetDict]]:
    """Load all DatasetDict give dataset keys

    Args:
        ds_keys_dict (Dict[str, List[str]]): A dictionary of dataset key list. {prediction_target: {list of ds_key (institution_phase)}}
        flag_more_feat_types (bool, optional): whether load all sensor types. Should be False for maximum compatibility. Defaults to False.
        verbose (bool, optional): Whether to display the progress bar and intermediate reuslts. Defaults to True

    Returns:
        Dict[str, Dict[str, DatasetDict]]: a dictionary of dictionary of DatasetDict. Level one is prediction_target, level two is ds_key
    """
    dataset_dict = {}

    for prediction_target, ds_keys in tqdm(ds_keys_dict.items(), position=0, desc= "loading prediction targets", disable=not verbose):
        for ds_key in tqdm(ds_keys, position=1, desc= "dataset keys", leave=False, disable=not verbose):
            if (verbose):
                tqdm.write("loading " + prediction_target + " " + ds_key + " " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            institution, phase = ds_key.split("_")
            phase = int(phase)
            if flag_more_feat_types: # currently only one institute can have max feature types
                assert institution in ["INS-W", "INS-W-sample"]
            dataset = data_loader_single(prediction_target, institution, phase, flag_more_feat_types)
            if (prediction_target not in dataset_dict):
                dataset_dict[prediction_target] = {}
            dataset_dict[prediction_target][ds_key] = dataset
        
    return dataset_dict
    
def data_loader_raw_single(institution:str, phase:int) -> pd.DataFrame:
    """Helper function to load raw data of a given institution and phase.
    If the data is already saved as pkl file, load the pkl file directly to accelerate the process.

    Args:
        institution (str): insitution code
        phase (int): number of study phase

    Returns:
        pd.DataFrame: raw data of a dataset
    """
    ds_key = f"{institution}_{phase}"
    dataset_file_path = os.path.join(path_definitions.DATA_PATH, "datarepo_df_raw", f"dep--{ds_key}.pkl")

    if (os.path.exists(dataset_file_path)):
        with open(dataset_file_path, "rb") as f:
            dataset_df = pickle.load(f)
    else:
        dataset_df = data_loader_single_dataset_raw(institution, phase, "dep_weekly")
        Path(os.path.split(dataset_file_path)[0]).mkdir(parents=True, exist_ok=True)
        with open(dataset_file_path, "wb") as f:
            pickle.dump(dataset_df, f)
    return dataset_df

def data_loader_raw(ds_keys_list: List[str], verbose:bool = True) -> Dict[str, pd.DataFrame]:
    """Load all raw data give dataset keys

    Args:
        ds_keys_list (List[str]): a list of dataset keys
        verbose (bool, optional): Whether to display the progress bar and intermediate reuslts. Defaults to True

    Returns:
        Dict[str, pd.DataFrame]: a dictionary of raw data, indexed by dataset keys
    """
    dataset_dict = {}

    for ds_key in tqdm(ds_keys_list, desc="loading dataset keys", disable= not verbose):
        if verbose:
            tqdm.write("loading " + ds_key + " " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        institution, phase = ds_key.split("_")
        phase = int(phase)
        dataset = data_loader_raw_single(institution, phase)
        dataset_dict[ds_key] = dataset
    return dataset_dict