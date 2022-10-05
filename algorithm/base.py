from typing import Union, List
from abc import ABC, abstractmethod
from data_loader.data_loader_ml import DataRepo, DatasetDict
from sklearn.base import BaseEstimator, ClassifierMixin

class DepressionDetectionClassifierBase(BaseEstimator, ClassifierMixin, ABC):
    """The basic depression detection classifier abstract template"""
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X:object, y:object):
        """Classifier training. Unique for each algorithm.

        Args:
            X (object): input data. Could be multiple data format
            y (object): label data
        """
        pass

    @abstractmethod
    def predict(self, X:object, y:object=None) -> List[object]:
        """Results prediction. Expected to return a list of labels

        Args:
            X (object): input data. Could be multiple data format
            y (object): label data - usually not needed. Default None
        """
        pass

    @abstractmethod
    def predict_proba(self, X:object, y:object=None) -> List[List[float]]:
        """Result probability prediction. Expected to return a list of probability distribution of all classes

        Args:
            X (object): input data. Could be multiple data format
            y (object): label data - usually not needed. Default None
        """
        pass

class DepressionDetectionAlgorithmBase(ABC):
    """ The basic depression detection algorithm abstract template """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def prep_data_repo(self, dataset:DatasetDict = None, flag_train:bool = True) -> DataRepo:
        """Prep a DataRepo class that contains X, y, and pid for model training/testing.
            Unique for each algorithm.

        Args:
            dataset (DatasetDict): dataset object that contains all features to be processed
            flag_train (bool): flag for whether the data repo is used for training and testing.
                This parameter may control some detail steps of preparation

        Returns:
            DataRepo: a prepared DataRepo object for model training and evaluation
        """
        pass

    @abstractmethod
    def prep_model(self) -> DepressionDetectionClassifierBase:
        """Prepare the depression detection classifier. Unique for each algorithm.

        Returns:
            DepressionDetectionClassifierBase: A depression detection classifier
        """
        pass