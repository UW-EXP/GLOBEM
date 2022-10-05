import os
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(Path(__file__).parent))

MODEL_PATH = os.path.join(PROJECT_ROOT, "model")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config")
UTILS_PATH = os.path.join(PROJECT_ROOT, "utils")
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
RAWDATA_PATH = os.path.join(PROJECT_ROOT, "data_raw")
TMP_PATH = os.path.join(PROJECT_ROOT, "tmp")
