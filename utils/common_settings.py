################
# Import commonly used packages
################

from typing import Dict, Tuple, List, Union
from pathlib import Path
import argparse
import glob
import random
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import math
import gc
import warnings
from copy import deepcopy
import json
import itertools
import collections
from datetime import datetime, timedelta
import time
import ast
import pickle
import sys
import traceback
import uuid
import yaml
import scipy
from scipy import stats
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from matplotlib.ticker import MultipleLocator
plt.style.use(['fivethirtyeight','ggplot'])
import seaborn as sns
from tqdm import tqdm
import warnings

# Set the maximum number of CPU to be used for multiprocessing
import multiprocessing
if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
    CPU_COUNT  = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
else:
    CPU_COUNT = multiprocessing.cpu_count()
NJOB = int(np.ceil(CPU_COUNT// 2))

from multiprocessing import Pool, Manager, set_start_method, get_context
from multiprocessing.pool import ThreadPool
import ray
import swifter
os.environ["MODIN_ENGINE"] = "ray"

import sklearn
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut,LeaveOneGroupOut,train_test_split, GroupShuffleSplit,GroupKFold, StratifiedGroupKFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, pairwise, pairwise_distances, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn import tree, svm, ensemble, linear_model
from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import GridSearchCV
import sklearn.cluster as cluster
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize

minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers, activations

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, Callback

from tensorflow.keras.layers import Layer, Input, Activation, Lambda, Flatten, Concatenate, add, Average
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Conv1D, Conv2D, ZeroPadding2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling1D, Cropping2D
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.layers import UpSampling1D, UpSampling2D, Reshape, Conv1DTranspose, Conv2DTranspose, InputSpec

from tensorflow.keras.initializers import glorot_uniform,he_uniform
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model,normalize, Sequence

sys.path.append(os.path.dirname(os.path.abspath(Path(__file__))))
from basic_utils import utils_ml
from basic_utils import utils_operation

################
# Ensure reproducibility
################

def set_random_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
seed = 42
set_random_seed(seed)
tf.keras.backend.set_floatx('float64')

################
# Define A few commonly used variables
################

wks = ["wkdy", "wkend"]
epochs_4 =  ["morning", "afternoon", "evening", "night"]
epochs_5 = epochs_4 + ["allday"]
epochs_6 = epochs_5 + ["14dhist"]

wks_epoch5 = [x + "_" + y for x in wks for y in epochs_5]
wks_epoch4 = [x + "_" + y for x in wks for y in epochs_4]
wkdy_epoch4 = ["wkdy_" + y for y in epochs_4]
wkend_epoch4 = ["wkend_" + y for y in epochs_4]
schema_defaults = ['pid', 'epoch', 'weekday', 'grouping', 'epoch_weekday_grouping_abbreviated', 'time']

feature_types = ['f_blue', 'f_call', 'f_screen', 'f_slp', 'f_steps', 'f_loc', 'f_locMap']

with open(os.path.join(os.path.dirname(os.path.abspath(Path(__file__).parent)),
    "config", f"global_config.yaml"), "r") as f:
    global_config = yaml.safe_load(f)

daterange_book = {
    "INS-W":{
        1:{
            "start_date" : "2018-04-03",
            "end_date"   : "2018-06-07"
        },
        2:{
            "start_date" : "2019-03-31",
            "end_date"   : "2019-06-15"
        },
        3:{
            "start_date" : "2020-03-30",
            "end_date"   : "2020-06-13"
        },
        4:{
            "start_date" : "2021-03-29",
            "end_date"   : "2021-06-12"
        },
    },

    "INS-D":{
        1:{
            "start_date" : "2018-03-26",
            "end_date"   : "2018-06-10"
        },
        2:{
            "start_date" : "2019-03-25",
            "end_date"   : "2019-06-09"
        }
    }
}

class feature_columns_repo:
    def __init__(self):
        self.feature_columns_fulldict = {}
        self.feature_columns_selecteddict = {}
        self.feature_columns_selected = []
        self.feature_columns_selected_dis = []
        self.feature_columns_selected_norm = []
        self.feature_columns_selected_allepoches = []
        self.feature_columns_selected_dis_allepoches = []
        self.feature_columns_selected_epoches = {}
        self.feature_columns_selected_epoches_types = {}

fc_repo = feature_columns_repo()

def set_feature_columns(sep = ":", epochs = epochs_5):
    """ Define a series of feature values in fc_repo """
    global fc_repo

    fc_repo.feature_columns_fulldict = {
        "f_blue": ['phone_bluetooth_rapids_countscans', 'phone_bluetooth_rapids_uniquedevices', 'phone_bluetooth_rapids_countscansmostuniquedevice', 'phone_bluetooth_doryab_countscansall', 'phone_bluetooth_doryab_uniquedevicesall', 'phone_bluetooth_doryab_meanscansall', 'phone_bluetooth_doryab_stdscansall', 'phone_bluetooth_doryab_countscansmostfrequentdevicewithinsegmentsall', 'phone_bluetooth_doryab_countscansmostfrequentdeviceacrosssegmentsall', 'phone_bluetooth_doryab_countscansmostfrequentdeviceacrossdatasetall', 'phone_bluetooth_doryab_countscansleastfrequentdevicewithinsegmentsall', 'phone_bluetooth_doryab_countscansleastfrequentdeviceacrosssegmentsall', 'phone_bluetooth_doryab_countscansleastfrequentdeviceacrossdatasetall', 'phone_bluetooth_doryab_countscansown', 'phone_bluetooth_doryab_uniquedevicesown', 'phone_bluetooth_doryab_meanscansown', 'phone_bluetooth_doryab_stdscansown', 'phone_bluetooth_doryab_countscansmostfrequentdevicewithinsegmentsown', 'phone_bluetooth_doryab_countscansmostfrequentdeviceacrosssegmentsown', 'phone_bluetooth_doryab_countscansmostfrequentdeviceacrossdatasetown', 'phone_bluetooth_doryab_countscansleastfrequentdevicewithinsegmentsown', 'phone_bluetooth_doryab_countscansleastfrequentdeviceacrosssegmentsown', 'phone_bluetooth_doryab_countscansleastfrequentdeviceacrossdatasetown', 'phone_bluetooth_doryab_countscansothers', 'phone_bluetooth_doryab_uniquedevicesothers', 'phone_bluetooth_doryab_meanscansothers', 'phone_bluetooth_doryab_stdscansothers', 'phone_bluetooth_doryab_countscansmostfrequentdevicewithinsegmentsothers', 'phone_bluetooth_doryab_countscansmostfrequentdeviceacrosssegmentsothers', 'phone_bluetooth_doryab_countscansmostfrequentdeviceacrossdatasetothers', 'phone_bluetooth_doryab_countscansleastfrequentdevicewithinsegmentsothers', 'phone_bluetooth_doryab_countscansleastfrequentdeviceacrosssegmentsothers', 'phone_bluetooth_doryab_countscansleastfrequentdeviceacrossdatasetothers'],
        "f_call": ['phone_calls_rapids_missed_count', 'phone_calls_rapids_missed_distinctcontacts', 'phone_calls_rapids_missed_timefirstcall', 'phone_calls_rapids_missed_timelastcall', 'phone_calls_rapids_missed_countmostfrequentcontact', 'phone_calls_rapids_incoming_count', 'phone_calls_rapids_incoming_distinctcontacts', 'phone_calls_rapids_incoming_meanduration', 'phone_calls_rapids_incoming_sumduration', 'phone_calls_rapids_incoming_minduration', 'phone_calls_rapids_incoming_maxduration', 'phone_calls_rapids_incoming_stdduration', 'phone_calls_rapids_incoming_modeduration', 'phone_calls_rapids_incoming_entropyduration', 'phone_calls_rapids_incoming_timefirstcall', 'phone_calls_rapids_incoming_timelastcall', 'phone_calls_rapids_incoming_countmostfrequentcontact', 'phone_calls_rapids_outgoing_count', 'phone_calls_rapids_outgoing_distinctcontacts', 'phone_calls_rapids_outgoing_meanduration', 'phone_calls_rapids_outgoing_sumduration', 'phone_calls_rapids_outgoing_minduration', 'phone_calls_rapids_outgoing_maxduration', 'phone_calls_rapids_outgoing_stdduration', 'phone_calls_rapids_outgoing_modeduration', 'phone_calls_rapids_outgoing_entropyduration', 'phone_calls_rapids_outgoing_timefirstcall', 'phone_calls_rapids_outgoing_timelastcall', 'phone_calls_rapids_outgoing_countmostfrequentcontact'],
        "f_loc": ["phone_locations_barnett_avgflightdur", "phone_locations_barnett_avgflightlen", "phone_locations_barnett_circdnrtn", "phone_locations_barnett_disttravelled", "phone_locations_barnett_hometime", "phone_locations_barnett_maxdiam", "phone_locations_barnett_maxhomedist", "phone_locations_barnett_probpause", "phone_locations_barnett_rog", "phone_locations_barnett_siglocentropy", "phone_locations_barnett_siglocsvisited", "phone_locations_barnett_stdflightdur", "phone_locations_barnett_stdflightlen", "phone_locations_barnett_wkenddayrtn", "phone_locations_doryab_avglengthstayatclusters", "phone_locations_doryab_avgspeed", "phone_locations_doryab_homelabel", "phone_locations_doryab_locationentropy", "phone_locations_doryab_locationvariance", "phone_locations_doryab_loglocationvariance", "phone_locations_doryab_maxlengthstayatclusters", "phone_locations_doryab_minlengthstayatclusters", "phone_locations_doryab_movingtostaticratio", "phone_locations_doryab_normalizedlocationentropy", "phone_locations_doryab_numberlocationtransitions", "phone_locations_doryab_numberofsignificantplaces", "phone_locations_doryab_outlierstimepercent", "phone_locations_doryab_radiusgyration", "phone_locations_doryab_stdlengthstayatclusters", "phone_locations_doryab_timeathome", "phone_locations_doryab_timeattop1location", "phone_locations_doryab_timeattop2location", "phone_locations_doryab_timeattop3location", "phone_locations_doryab_totaldistance", "phone_locations_doryab_varspeed", 'phone_locations_locmap_duration_in_locmap_study', 'phone_locations_locmap_percent_in_locmap_study', 'phone_locations_locmap_duration_in_locmap_exercise', 'phone_locations_locmap_percent_in_locmap_exercise', 'phone_locations_locmap_duration_in_locmap_greens', 'phone_locations_locmap_percent_in_locmap_greens'],
        "f_screen": ['phone_screen_rapids_countepisodeunlock', 'phone_screen_rapids_sumdurationunlock', 'phone_screen_rapids_maxdurationunlock', 'phone_screen_rapids_mindurationunlock', 'phone_screen_rapids_avgdurationunlock', 'phone_screen_rapids_stddurationunlock', 'phone_screen_rapids_firstuseafter00unlock', 'phone_screen_rapids_countepisodeunlock_locmap_exercise', 'phone_screen_rapids_sumdurationunlock_locmap_exercise', 'phone_screen_rapids_maxdurationunlock_locmap_exercise', 'phone_screen_rapids_mindurationunlock_locmap_exercise', 'phone_screen_rapids_avgdurationunlock_locmap_exercise', 'phone_screen_rapids_stddurationunlock_locmap_exercise', 'phone_screen_rapids_firstuseafter00unlock_locmap_exercise', 'phone_screen_rapids_countepisodeunlock_locmap_greens', 'phone_screen_rapids_sumdurationunlock_locmap_greens', 'phone_screen_rapids_maxdurationunlock_locmap_greens', 'phone_screen_rapids_mindurationunlock_locmap_greens', 'phone_screen_rapids_avgdurationunlock_locmap_greens', 'phone_screen_rapids_stddurationunlock_locmap_greens', 'phone_screen_rapids_firstuseafter00unlock_locmap_greens', 'phone_screen_rapids_countepisodeunlock_locmap_living', 'phone_screen_rapids_sumdurationunlock_locmap_living', 'phone_screen_rapids_maxdurationunlock_locmap_living', 'phone_screen_rapids_mindurationunlock_locmap_living', 'phone_screen_rapids_avgdurationunlock_locmap_living', 'phone_screen_rapids_stddurationunlock_locmap_living', 'phone_screen_rapids_firstuseafter00unlock_locmap_living', 'phone_screen_rapids_countepisodeunlock_locmap_study', 'phone_screen_rapids_sumdurationunlock_locmap_study', 'phone_screen_rapids_maxdurationunlock_locmap_study', 'phone_screen_rapids_mindurationunlock_locmap_study', 'phone_screen_rapids_avgdurationunlock_locmap_study', 'phone_screen_rapids_stddurationunlock_locmap_study', 'phone_screen_rapids_firstuseafter00unlock_locmap_study', 'phone_screen_rapids_countepisodeunlock_locmap_home', 'phone_screen_rapids_sumdurationunlock_locmap_home', 'phone_screen_rapids_maxdurationunlock_locmap_home', 'phone_screen_rapids_mindurationunlock_locmap_home', 'phone_screen_rapids_avgdurationunlock_locmap_home', 'phone_screen_rapids_stddurationunlock_locmap_home', 'phone_screen_rapids_firstuseafter00unlock_locmap_home'],
        "f_slp": ['fitbit_sleep_summary_rapids_sumdurationafterwakeupmain', 'fitbit_sleep_summary_rapids_sumdurationasleepmain', 'fitbit_sleep_summary_rapids_sumdurationawakemain', 'fitbit_sleep_summary_rapids_sumdurationtofallasleepmain', 'fitbit_sleep_summary_rapids_sumdurationinbedmain', 'fitbit_sleep_summary_rapids_avgefficiencymain', 'fitbit_sleep_summary_rapids_avgdurationafterwakeupmain', 'fitbit_sleep_summary_rapids_avgdurationasleepmain', 'fitbit_sleep_summary_rapids_avgdurationawakemain', 'fitbit_sleep_summary_rapids_avgdurationtofallasleepmain', 'fitbit_sleep_summary_rapids_avgdurationinbedmain', 'fitbit_sleep_summary_rapids_countepisodemain', 'fitbit_sleep_summary_rapids_firstbedtimemain', 'fitbit_sleep_summary_rapids_lastbedtimemain', 'fitbit_sleep_summary_rapids_firstwaketimemain', 'fitbit_sleep_summary_rapids_lastwaketimemain', 'fitbit_sleep_intraday_rapids_avgdurationasleepunifiedmain', 'fitbit_sleep_intraday_rapids_avgdurationawakeunifiedmain', 'fitbit_sleep_intraday_rapids_maxdurationasleepunifiedmain', 'fitbit_sleep_intraday_rapids_maxdurationawakeunifiedmain', 'fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain', 'fitbit_sleep_intraday_rapids_sumdurationawakeunifiedmain', 'fitbit_sleep_intraday_rapids_countepisodeasleepunifiedmain', 'fitbit_sleep_intraday_rapids_countepisodeawakeunifiedmain', 'fitbit_sleep_intraday_rapids_stddurationasleepunifiedmain', 'fitbit_sleep_intraday_rapids_stddurationawakeunifiedmain', 'fitbit_sleep_intraday_rapids_mindurationasleepunifiedmain', 'fitbit_sleep_intraday_rapids_mindurationawakeunifiedmain', 'fitbit_sleep_intraday_rapids_mediandurationasleepunifiedmain', 'fitbit_sleep_intraday_rapids_mediandurationawakeunifiedmain', 'fitbit_sleep_intraday_rapids_ratiocountasleepunifiedwithinmain', 'fitbit_sleep_intraday_rapids_ratiocountawakeunifiedwithinmain', 'fitbit_sleep_intraday_rapids_ratiodurationasleepunifiedwithinmain', 'fitbit_sleep_intraday_rapids_ratiodurationawakeunifiedwithinmain'],
        "f_steps": ['fitbit_steps_summary_rapids_maxsumsteps', 'fitbit_steps_summary_rapids_minsumsteps', 'fitbit_steps_summary_rapids_avgsumsteps', 'fitbit_steps_summary_rapids_mediansumsteps', 'fitbit_steps_summary_rapids_stdsumsteps', 'fitbit_steps_intraday_rapids_sumsteps', 'fitbit_steps_intraday_rapids_maxsteps', 'fitbit_steps_intraday_rapids_minsteps', 'fitbit_steps_intraday_rapids_avgsteps', 'fitbit_steps_intraday_rapids_stdsteps', 'fitbit_steps_intraday_rapids_countepisodesedentarybout', 'fitbit_steps_intraday_rapids_sumdurationsedentarybout', 'fitbit_steps_intraday_rapids_maxdurationsedentarybout', 'fitbit_steps_intraday_rapids_mindurationsedentarybout', 'fitbit_steps_intraday_rapids_avgdurationsedentarybout', 'fitbit_steps_intraday_rapids_stddurationsedentarybout', 'fitbit_steps_intraday_rapids_countepisodeactivebout', 'fitbit_steps_intraday_rapids_sumdurationactivebout', 'fitbit_steps_intraday_rapids_maxdurationactivebout', 'fitbit_steps_intraday_rapids_mindurationactivebout', 'fitbit_steps_intraday_rapids_avgdurationactivebout', 'fitbit_steps_intraday_rapids_stddurationactivebout'],
    }
    fc_repo.feature_columns_selecteddict = fc_repo.feature_columns_fulldict
            

    fc_repo.feature_columns_selected_origin = []
    fc_repo.feature_columns_selected = []
    for f, fl in fc_repo.feature_columns_selecteddict.items():
        fc_repo.feature_columns_selected_origin += fl
        fc_repo.feature_columns_selected += [f + sep + x for x in fl]
        
    fc_repo.feature_columns_selected_dis = []
    for f, fl in fc_repo.feature_columns_selecteddict.items():
        fc_repo.feature_columns_selected_dis += [f + sep + x + "_dis" for x in fl]

    fc_repo.feature_columns_selected_norm = []
    for f, fl in fc_repo.feature_columns_selecteddict.items():
        fc_repo.feature_columns_selected_norm += [f + sep + x + "_norm" for x in fl]
        
    fc_repo.feature_columns_selected_allepoches = []
    for f in fc_repo.feature_columns_selected:
        fc_repo.feature_columns_selected_allepoches += [f + sep + x for x in epochs]
        
    fc_repo.feature_columns_selected_dis_allepoches = []
    for f in fc_repo.feature_columns_selected_dis:
        fc_repo.feature_columns_selected_dis_allepoches += [f + sep + x for x in epochs]


    fc_repo.feature_columns_selected_epoches = {}
    fc_repo.feature_columns_selected_types = {}
    fc_repo.feature_columns_selected_epoches_types = {}

    for epoch in epochs:
        fc_repo.feature_columns_selected_epoches[epoch] = [f + sep + epoch for f in fc_repo.feature_columns_selected]
        fc_repo.feature_columns_selected_epoches_types[epoch] = {}
        for f in fc_repo.feature_columns_selected:
            ft = f.split(sep)[0]
            if (ft in fc_repo.feature_columns_selected_epoches_types[epoch]):
                fc_repo.feature_columns_selected_epoches_types[epoch][ft].append(f + sep + epoch)
            else:
                fc_repo.feature_columns_selected_epoches_types[epoch][ft] = [f + sep + epoch]
    for f in fc_repo.feature_columns_selected:
        ft = f.split(sep)[0]
        if (ft in fc_repo.feature_columns_selected_types):
            fc_repo.feature_columns_selected_types[ft] += [f + sep + epoch for epoch in epochs]
        else:
            fc_repo.feature_columns_selected_types[ft] = [f + sep + epoch for epoch in epochs]
                
    fc_repo.feature_columns_selected_dis_epoches = {}
    fc_repo.feature_columns_selected_dis_epoches_types = {}

    for epoch in epochs:
        fc_repo.feature_columns_selected_dis_epoches[epoch] = [f + sep + epoch for f in fc_repo.feature_columns_selected_dis]
        fc_repo.feature_columns_selected_dis_epoches_types[epoch] = {}
        for f in fc_repo.feature_columns_selected_dis:
            ft = f.split(sep)[0]
            if (ft in fc_repo.feature_columns_selected_dis_epoches_types[epoch]):
                fc_repo.feature_columns_selected_dis_epoches_types[epoch][ft].append(f + sep + epoch)
            else:
                fc_repo.feature_columns_selected_dis_epoches_types[epoch][ft] = [f + sep + epoch]

    fc_repo.feature_columns_selected_norm_epoches = {}
    fc_repo.feature_columns_selected_norm_epoches_types = {}

    for epoch in epochs:
        fc_repo.feature_columns_selected_norm_epoches[epoch] = [f + sep + epoch for f in fc_repo.feature_columns_selected_norm]
        fc_repo.feature_columns_selected_norm_epoches_types[epoch] = {}
        for f in fc_repo.feature_columns_selected_norm:
            ft = f.split(sep)[0]
            if (ft in fc_repo.feature_columns_selected_norm_epoches_types[epoch]):
                fc_repo.feature_columns_selected_norm_epoches_types[epoch][ft].append(f + sep + epoch)
            else:
                fc_repo.feature_columns_selected_norm_epoches_types[epoch][ft] = [f + sep + epoch]

set_feature_columns(sep = ":", epochs=epochs_6)

def globalize(func):
    """Put a function into the global environment so that multiprocessing is easier"""
    def result(*args, **kwargs):
      return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result

def get_min_count_class(labels, groups):
    df_tmp = pd.DataFrame([labels,groups]).T
    df_tmp.columns = ["label", "group"]
    num_min = df_tmp.groupby("label").apply(lambda x : len(set(x["group"]))).min()
    return num_min