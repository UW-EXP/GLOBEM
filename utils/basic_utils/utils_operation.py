from typing import Dict
import numpy as np
import pandas as pd
from copy import deepcopy
import time
from sklearn.preprocessing import RobustScaler

def get_dict_element(d: Dict) -> np.ndarray:
    """Get the element of a dict as a numpy list
    
    Args:
        d (dictionary)
    
    Returns:
        np.ndarray: the list of the value of all keys in the dict
    """
    return np.concatenate([v for k,v in d.items()])

def get_df_rows(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Get a subset of the rules by some columns
    
    Args:
        df (pd.Dataframe): A dataframe to be extracted
        parameters (dictionary): {col:value (can be a single or a list of value)}
    
    Returns:
        pd.Dataframe: A dataframe with the selected rows
    """
    for col, val in parameters.items():
        if (type(val) is list):
            df = df[df[col].isin(val)]
        else:
            df = df[df[col] == val]
    return df

def discretize_tertiles(df: pd.DataFrame) -> pd.DataFrame:
    """Split the data into one to three levels

    Args:
        df (pd.DataFrame): Data frame

    Returns:
        pd.DataFrame: The data after discretization
    """
    df_buf = deepcopy(df)
    for col in df.columns:
        try:
            df_buf[col] = pd.qcut(df[col], q=3, labels=["l", "m", "h"])
        except Exception:
            try:
                # Cases where there are only two unique values
                df_buf[col] = pd.qcut(df[col], q=2, labels=["l", "h"])
            except Exception:
                # Cases where everything is the same value
                df_buf[col] = pd.qcut(df[col], q=1, labels=["m"])
    return df_buf

def discretize_data(df:pd.DataFrame, grpbyid:str) -> pd.DataFrame:
    """discretize real-value data into three levels, based on certain group
    
    Args:
        df (pandas dataframe): data
        grpbyid (string): groupby column 
    
    Returns:
        pd.DataFrame: data after descretization
    """

    # from pandarallel import pandarallel
    # pandarallel.initialize(progress_bar = False, nb_workers = 8, verbose = 1)
    features = list(df.columns)
    features_dis = [f for f in features if df[f].dtype.kind in "bifc"]
    df_dis = df.groupby(grpbyid).apply(lambda x: discretize_tertiles(
        x[features_dis])).reset_index(level=-1, drop=True)
    # del pandarallel
    return df_dis

def normalize_robust(df:pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """Normalize the data using quantile, robust to outliers

    Args:
        df (pd.DataFrame): data to be normalized
        scaler (RobustScaler): sklearn.preprocessing.robustscaler

    Returns:
        pd.DataFrame: data after normalization
    """
    df_buf = deepcopy(df)
    na_count = df_buf.isna().sum(axis = 0).values
    notfullna_idx = np.where(na_count < df_buf.shape[0])[0]
    cols = list(df_buf.columns[notfullna_idx])
    scl = scaler.fit(df_buf[cols])
    df_buf[cols] = scl.transform(df_buf[cols]) # [5%,95%] to [-1,1]
    df_buf[cols] = df_buf[cols].clip(lower = -2, upper = 2) # reduce outlier effect
    return df_buf


def normalize_data(df, grpbyid, scaler = None) -> pd.DataFrame:
    """normalize real-value data, based on certain group
    
    Args:
        df (pandas dataframe): data
        grpbyid (string): groupby column 
    
    Returns:
        pd.DataFrame: data after normalization
    """
    if scaler is None:
        scaler = RobustScaler(quantile_range = (5,95), unit_variance = False)
    features = list(df.columns)
    features_norm = [f for f in features if df[f].dtype.kind in "bifc"]
    df_norm = df.groupby(grpbyid).apply(lambda x: normalize_robust(
        x[features_norm],scaler)).reset_index(level=-1, drop=True)
    return df_norm


def func_timer(func, repeat = 10):
    """Time a function

    Args:
        func : The target function
        repeat (int, optional): the number of repetition. Defaults to 10.
    """
    start = time.time()
    for _ in range(repeat):
        func()
    end = time.time()
    print("avg time: ", (end - start) / repeat)

def isnotebook() -> bool:
    """judge whether the script is ran in a notebook

    Returns:
        bool: whether it is in notebook
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter