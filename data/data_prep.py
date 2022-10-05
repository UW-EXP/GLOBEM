import os, yaml, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from data_loader import data_loader_ml, data_loader_dl
from utils import path_definitions
from data import data_factory
import wget
import zipfile

def download_rawsampledata(key:str = "data_raw_sample") -> None:
    """ Download and unzip the data """

    target_path = os.path.join(path_definitions.RAWDATA_PATH, key + ".zip")
    if (os.path.exists(target_path)):
        return
        
    url = data_factory.url_dictionary[key]
    print("Downloading sample data...")
    wget.download(url, out = target_path)
    print("Unzipping data...")
    with zipfile.ZipFile(target_path,"r") as zip_ref:
        zip_ref.extractall(path_definitions.RAWDATA_PATH)
    print("Unzipping done!")


def convert_rawdata_to_pkldata() -> None:
    """ Convert raw data from 'data_raw' folder to pkl data in 'data' folder """

    with open(os.path.join(os.path.dirname(os.path.abspath(Path(__file__).parent)),
        "config", f"global_config.yaml"), "r") as f:
        global_config = yaml.safe_load(f)

    ds_keys = global_config["all"]["ds_keys"]
    pred_targets = global_config["all"]["prediction_tasks"]

    ds_keys_dict = {
        pred_target: ds_keys for pred_target in pred_targets
    }
    
    # prepare pkl file with raw time-series format, only used for non-ML analysis purpose.
    # Currently only support dep_weekly
    if ("dep_weekly" in pred_targets):
        data_loader_ml.data_loader_raw(ds_keys_list=ds_keys)
    
    if not global_config["all"]["flag_more_feat_types"]:
        # prepare pkl file with sliced data format
        data_loader_ml.data_loader(
            ds_keys_dict={pt: ds_keys for pt in pred_targets}, flag_more_feat_types=False)
        # prepare np pkl file with sliced data format for deep models
        data_loader_dl.data_loader_np(ds_keys_dict=ds_keys_dict, flag_normalize=True, flag_more_feat_types=False)
    else: # prepare pkl file with all sensor types (currently support INS-W)
        # prepare pkl file with sliced data format
        data_loader_ml.data_loader(
            ds_keys_dict={pt: ds_keys for pt in pred_targets}, flag_more_feat_types=True)
        # prepare np pkl file with sliced data format for deep models
        data_loader_dl.data_loader_np(ds_keys_dict=ds_keys_dict, flag_normalize=True, flag_more_feat_types=True)

    # prepare placeholder to make sure the pipeline will go smoothly for deep models
    data_loader_dl.data_loader_dl_placeholder(pred_targets, ds_keys)

if __name__ == "__main__":

    convert_rawdata_to_pkldata()