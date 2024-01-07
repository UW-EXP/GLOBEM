import os, sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import path_definitions

def get_raw_feature_folderpath(phase = 1, institute = "INS"):
    path = os.path.join(path_definitions.RAWDATA_PATH , institute + "_" + str(phase) , "FeatureData/")
    return path

def get_survey_filepath(phase = 1, institute = "INS"):
    path = os.path.join(path_definitions.RAWDATA_PATH , institute + "_" + str(phase) , "SurveyData/")
    return path

def get_participantsinfo_filepath(phase = 1, institute = "INS"):
    path = os.path.join(path_definitions.RAWDATA_PATH , institute + "_" + str(phase) , "ParticipantsInfoData/")
    return path

feature_folder = {
    "INS-W": {1:get_raw_feature_folderpath(1, "INS-W"),
        2:get_raw_feature_folderpath(2, "INS-W"),
        3:get_raw_feature_folderpath(3, "INS-W"),
        4:get_raw_feature_folderpath(4, "INS-W")},
    "INS-W-sample": {1:get_raw_feature_folderpath(1, "INS-W-sample"),
        2:get_raw_feature_folderpath(2, "INS-W-sample"),
        3:get_raw_feature_folderpath(3, "INS-W-sample"),
        4:get_raw_feature_folderpath(4, "INS-W-sample")},
    "INS-D": {1:get_raw_feature_folderpath(1, "INS-D"),
        2: get_raw_feature_folderpath(2, "INS-D")},
}

survey_folder = {
    "INS-W": {1:get_survey_filepath(1, "INS-W"),
        2:get_survey_filepath(2, "INS-W"),
        3:get_survey_filepath(3, "INS-W"),
        4:get_survey_filepath(4, "INS-W"),},
    "INS-W-sample": {1:get_survey_filepath(1, "INS-W-sample"),
        2:get_survey_filepath(2, "INS-W-sample"),
        3:get_survey_filepath(3, "INS-W-sample"),
        4:get_survey_filepath(4, "INS-W-sample"),},
    "INS-D": {1:get_survey_filepath(1, "INS-D"), 
        2: get_survey_filepath(2, "INS-D")},
}

participants_info_folder = {
    "INS-W": {1:get_participantsinfo_filepath(1, "INS-W"),
        2:get_participantsinfo_filepath(2, "INS-W"),
        3:get_participantsinfo_filepath(3, "INS-W"),
        4:get_participantsinfo_filepath(4, "INS-W")},
    "INS-W-sample": {1:get_participantsinfo_filepath(1, "INS-W-sample"),
        2:get_participantsinfo_filepath(2, "INS-W-sample"),
        3:get_participantsinfo_filepath(3, "INS-W-sample"),
        4:get_participantsinfo_filepath(4, "INS-W-sample")},
    "INS-D": {1:get_participantsinfo_filepath(1, "INS-D"),
        2:get_participantsinfo_filepath(2, "INS-D")},
}

url_dictionary = {
    "data_raw_sample": "https://drive.google.com/uc?export=download&id=1a3cM1joYyPPoYmDCk1U4qM3Q_gjSeJLd"
}

# Some preliminary thresholds for initial testing are based on the split percentile.
# Open to change if there is theoretical support
threshold_book = {
    "UCLA_10items_POST": {"threshold_as_false": 24, "threshold_as_true":25},
    "negative_affect_EMA": {"threshold_as_false": 5, "threshold_as_true":6},
    "PSS_10items_POST": {"threshold_as_false": 20, "threshold_as_true":21},
    "CHIPS_POST":  {"threshold_as_false": 16, "threshold_as_true":17},
    "SocialFit_POST":  {"threshold_as_false": 76, "threshold_as_true":77},
    "ERQ_reappraisal_POST": {"threshold_as_false": 4.9, "threshold_as_true":5.0},
    "ERQ_suppression_POST": {"threshold_as_false": 4.9, "threshold_as_true":5.0},
    "BRS_POST": {"threshold_as_false": 2.99, "threshold_as_true":3.0},
    "2waySSS_giving_emotional_POST": {"threshold_as_false": 19.9, "threshold_as_true":20.0},
    "2waySSS_giving_instrumental_POST": {"threshold_as_false": 16.9, "threshold_as_true":17.0},
    "2waySSS_receiving_emotional_POST": {"threshold_as_false": 29.9, "threshold_as_true":30.0},
    "2waySSS_receiving_instrumental_POST": {"threshold_as_false": 16.5, "threshold_as_true":16.6},
    "phq4_anxiety_EMA": {"threshold_as_false": 2, "threshold_as_true":3},
    "pss4_EMA": {"threshold_as_false": 7, "threshold_as_true":8},
    "positive_affect_EMA": {"threshold_as_false": 5, "threshold_as_true":6},
}
