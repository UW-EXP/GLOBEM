import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".." )))

from algorithm.base import DepressionDetectionAlgorithmBase
from algorithm.ml_saeb import DepressionDetectionAlgorithm_ML_saeb
from algorithm.ml_canzian import DepressionDetectionAlgorithm_ML_canzian
from algorithm.ml_wahle import DepressionDetectionAlgorithm_ML_wahle
from algorithm.ml_farhan import DepressionDetectionAlgorithm_ML_farhan
from algorithm.ml_lu import DepressionDetectionAlgorithm_ML_lu
from algorithm.ml_wang import DepressionDetectionAlgorithm_ML_wang
from algorithm.ml_chikersal import DepressionDetectionAlgorithm_ML_chikersal
from algorithm.ml_xu_interpretable import DepressionDetectionAlgorithm_ML_xu_interpretable
from algorithm.ml_xu_personalized import DepressionDetectionAlgorithm_ML_xu_personalized

from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm
from algorithm.dl_siamese import DepressionDetectionAlgorithm_DL_siamese
from algorithm.dl_reorder import DepressionDetectionAlgorithm_DL_reorder
from algorithm.dl_mldg import DepressionDetectionAlgorithm_DL_mldg
from algorithm.dl_masf import DepressionDetectionAlgorithm_DL_masf
from algorithm.dl_csd import DepressionDetectionAlgorithm_DL_csd
from algorithm.dl_dann import DepressionDetectionAlgorithm_DL_dann
from algorithm.dl_irm import DepressionDetectionAlgorithm_DL_irm
from algorithm.dl_clustering import DepressionDetectionAlgorithm_DL_clustering

def load_algorithm(config_name:str) -> DepressionDetectionAlgorithmBase:
    """Load an algorithm given a config name

    Args:
        config_name (str): config name

    Raises:
        ValueError: Unsupported algorithm

    Returns:
        DepressionDetectionAlgorithmBase: algorithm object
    """
    if (config_name.startswith("ml_saeb")):
        algorithm = DepressionDetectionAlgorithm_ML_saeb(config_name = config_name)
    elif (config_name.startswith("ml_canzian")):
        algorithm = DepressionDetectionAlgorithm_ML_canzian(config_name = config_name)
    elif (config_name.startswith("ml_wahle")):
        algorithm = DepressionDetectionAlgorithm_ML_wahle(config_name = config_name)
    elif (config_name.startswith("ml_farhan")):
        algorithm = DepressionDetectionAlgorithm_ML_farhan(config_name = config_name)
    elif (config_name.startswith("ml_lu")):
        algorithm = DepressionDetectionAlgorithm_ML_lu(config_name = config_name)
    elif (config_name.startswith("ml_wang")):
        algorithm = DepressionDetectionAlgorithm_ML_wang(config_name = config_name)
    elif (config_name.startswith("ml_chikersal")):
        algorithm = DepressionDetectionAlgorithm_ML_chikersal(config_name = config_name)
    elif (config_name.startswith("ml_xu_interpretable")):
        algorithm = DepressionDetectionAlgorithm_ML_xu_interpretable(config_name = config_name)
    elif (config_name.startswith("ml_xu_personalized")):
        algorithm = DepressionDetectionAlgorithm_ML_xu_personalized(config_name = config_name)
    elif (config_name.startswith("dl_erm")):
        algorithm = DepressionDetectionAlgorithm_DL_erm(config_name = config_name)
    elif (config_name.startswith("dl_irm")):
        algorithm = DepressionDetectionAlgorithm_DL_irm(config_name = config_name)
    elif (config_name.startswith("dl_mldg")):
        algorithm = DepressionDetectionAlgorithm_DL_mldg(config_name = config_name)
    elif (config_name.startswith("dl_masf")):
        algorithm = DepressionDetectionAlgorithm_DL_masf(config_name = config_name)
    elif (config_name.startswith("dl_dann")):
        algorithm = DepressionDetectionAlgorithm_DL_dann(config_name = config_name)
    elif (config_name.startswith("dl_csd")):
        algorithm = DepressionDetectionAlgorithm_DL_csd(config_name = config_name)
    elif (config_name.startswith("dl_siamese")):
        algorithm = DepressionDetectionAlgorithm_DL_siamese(config_name = config_name)
    elif ("dl_reorder" in config_name):
        algorithm = DepressionDetectionAlgorithm_DL_reorder(config_name = config_name)
    elif ("dl_clustering" in config_name):
        algorithm = DepressionDetectionAlgorithm_DL_clustering(config_name = config_name)
    else:
        raise ValueError(f"config_name: {config_name} is not supported")

    return algorithm