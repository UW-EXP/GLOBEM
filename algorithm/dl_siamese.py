"""
Implementation of metric-learning based Siamese network for depression detection

reference:

Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. 2015. 
Siamese Neural Networks for One-shot Image Recognition.
Proceedings of the 32nd International Conference on Machine Learning (2015), 8.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm, DepressionDetectionClassifier_DL_erm
from data_loader.data_loader_ml import DataRepo
from utils import network, tf_metric_loss_64bit

class EvaluationCallback_siamese(network.EvaluationBasicCallback):
    """ Siamese network evaluation callback function """
    
    def __init__(self, model_obj, metrics_params, dataset_train, dataset_test=None, interval=1, verbose=1):
        super().__init__(model_obj, dataset_train, dataset_test, interval, verbose, flag_skip_y_defition=True)
        for d, l in self.dataset_train:
            self.X_train = d
            self.y_train = np.array(l)
                
        if (self.flag_with_test):
            for d, l in self.dataset_test:
                self.X_test = d
                self.y_test = np.array(l)
        
        self.metrics_params = [metrics_params]

    def judge_via_neighbor(self, distance_matrix_train, y_train, distance_matrix_test, y_test,
                            n_neighbor, positive_weight_ratio = 1.0, direction = 'both', flag_distance_weight=False):

        y_pred_train, y_pred_prob_train = pred_via_neighbor(distance_matrix_train, y_train=y_train,
                                        n_neighbor=n_neighbor, positive_weight_ratio=positive_weight_ratio, is_train = True,
                                        direction=direction, flag_distance_weight=flag_distance_weight)

        results_train = utils_ml.results_report(y_test=y_train, y_pred = y_pred_train, return_confusion_mtx=True, verbose=False)
        try:
            results_train["roc_auc"] = roc_auc_score(y_true=y_train, y_score = y_pred_prob_train[:,1])
        except:
            results_train["roc_auc"] = np.nan
        if (self.flag_with_test):
            y_pred_test, y_pred_prob_test = pred_via_neighbor(distance_matrix_test, y_train=y_train,
                                            n_neighbor=n_neighbor, positive_weight_ratio=positive_weight_ratio,
                                            direction=direction, flag_distance_weight=flag_distance_weight)
            results_test = utils_ml.results_report(y_test=y_test, y_pred = y_pred_test, return_confusion_mtx=True, verbose=False)
            try:
                results_test["roc_auc"] = roc_auc_score(y_true=y_test, y_score = y_pred_prob_test[:,1])
            except:
                results_test["roc_auc"] = np.nan
        else:
            results_test = None
        return results_train, results_test

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.interval == 0):
            embeddings_train = self.model_obj.clf.predict(self.X_train)
            distance_matrix_train = compute_distance_matrix(embeddings_train=embeddings_train,
                embeddings_test=embeddings_train)

            if (self.flag_with_test):
                embeddings_test = self.model_obj.clf.predict(self.X_test)
                distance_matrix_test = compute_distance_matrix(embeddings_train=embeddings_train,
                    embeddings_test=embeddings_test)
            else:
                distance_matrix_test = None
                self.y_test = None
            
            for params in self.metrics_params:
                results_train, results_test = self.judge_via_neighbor(
                    distance_matrix_train, self.y_train,
                    distance_matrix_test, self.y_test,
                    n_neighbor=params["n_neighbor"],
                    positive_weight_ratio=params["positive_weight_ratio"],
                    direction=params["direction"],
                    flag_distance_weight=params["flag_distance_weight"]
                )
                logs_new = deepcopy(logs)
                logs_new.update(params)
                self.process_results(epoch, logs_new, results_train, results_test)

class DepressionDetectionClassifier_DL_siamese(DepressionDetectionClassifier_DL_erm):
    """ Siamese network, extended from ERM classifier """

    def __init__(self, config):
        super().__init__(config=config)
        self.metrics_params = self.config["metrics_params"]

    def prep_eval_callbacks(self, X):
        ds_train = X["val_whole"]
        if "test" in X:
            ds_test = X["test"]
        elif "val" in X:
            ds_test = X["val"]
        else:
            ds_test = None
        return EvaluationCallback_siamese(model_obj=self, metrics_params=self.metrics_params,
                dataset_train=ds_train, dataset_test=ds_test,
                interval=1, verbose=self.training_params["verbose"])

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(42)

        self.__assert__(X)

        model_optimizer = network.prep_model_optimizer(self.training_params)

        self.clf.compile(loss = tf_metric_loss_64bit.TripletSemiHardLoss_64bit(
                margin=self.model_params["triplet_loss_margin"]),
            optimizer = model_optimizer)

        callbacks = self.prep_callbacks(X)

        if (self.training_params.get("skip_training", False) == False):

            for d_val, l_val in (X["val"] if self.flag_X_dict else X): break
            
            history = self.clf.fit(x = X["train"] if self.flag_X_dict else X,
                    steps_per_epoch = self.training_params["steps_per_epoch"],
                    epochs = self.training_params["epochs"],
                    validation_data = (d_val, l_val),
                    verbose = 1 if self.training_params["verbose"] > 1 else 0,
                    callbacks = callbacks
                    )

            self.log_history = history.history

            best_epoch, df_results_record = self.find_best_epoch()

            self.clf.set_weights(self.model_saver.model_repo_dict[best_epoch])

        else:
            df_results_record = self.fit_skip_training()

        self.embeddings_train_repo = self.clf.predict(X["val_whole"] if self.flag_X_dict else X)
        for d, l in X["val_whole"]:
            self.y_train_repo = np.array(l)

        return df_results_record

    def single_predict(self, clf, data, embeddings_train_repo, y_train_repo, flag_is_train):
        embeddings_test = clf.predict(data)
        distance_matrix = compute_distance_matrix(embeddings_train=embeddings_train_repo,
            embeddings_test=embeddings_test)
        y_pred, y_pred_prob = pred_via_neighbor(distance_matrix, y_train=y_train_repo,
                n_neighbor=self.metrics_params["n_neighbor"],
                positive_weight_ratio=self.metrics_params["positive_weight_ratio"],
                direction=self.metrics_params["direction"],
                flag_distance_weight=self.metrics_params["flag_distance_weight"],
                is_train=flag_is_train)
        return y_pred, y_pred_prob

    def predict(self, X, y=None):
        self.__assert__(X)
        if (self.flag_X_dict):
            X_ = X["val_whole"] # only use the whole val set for eval
            flag_is_train = True
        else:
            X_ = X
            flag_is_train = False
        for data, label in X_:
            y_pred, y_pred_prob = self.single_predict(self.clf, data,
                self.embeddings_train_repo, self.y_train_repo, flag_is_train)
            return y_pred

    def predict_proba(self, X, y=None):
        self.__assert__(X)
        if (self.flag_X_dict):
            X_ = X["val_whole"] # only use the whole val set for eval
            flag_is_train = True
        else:
            flag_is_train = False
            X_ = X
        for data, label in X_:
            y_pred, y_pred_prob = self.single_predict(self.clf, data,
                self.embeddings_train_repo, self.y_train_repo, flag_is_train)
            return y_pred_prob

class DepressionDetectionAlgorithm_DL_siamese(DepressionDetectionAlgorithm_DL_erm):
    """ The Siamese network algorithm. Extends the ERM algorithm """

    def __init__(self, config_dict = None, config_name = "dl_siamese"):
        super().__init__(config_dict, config_name)
    
    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        self.config["model_params"].update(
            {"input_shape": self.input_shape,
            "flag_return_embedding":True, "flag_embedding_norm":True,
            "flag_input_dict":True}
        )
        return DepressionDetectionClassifier_DL_siamese(config=self.config)

# post-process functions

def compute_distance_matrix(embeddings_train: np.ndarray, embeddings_test: np.ndarray) -> np.ndarray:
    """Compute pairwise distance on embedding matrix"""
    # For each pics of our dataset
    try:
        distance_matrix = pairwise_distances(X=embeddings_train,
                            Y=embeddings_test, metric="euclidean")
    except:
        embeddings_train = np.nan_to_num(embeddings_train, np.mean(embeddings_train))
        embeddings_test = np.nan_to_num(embeddings_test, np.mean(embeddings_test))
        embeddings_train = np.clip(embeddings_train, a_min = -1e9, a_max = 1e9)
        embeddings_test = np.clip(embeddings_test, a_min = -1e9, a_max = 1e9)
        distance_matrix = pairwise_distances(X=embeddings_train,
                            Y=embeddings_test, metric="euclidean")
    return distance_matrix


def pred_via_neighbor(distance_matrix: np.ndarray, y_train: np.ndarray,
                       n_neighbor:int, positive_weight_ratio:float = 1.0, direction:str = 'both', flag_distance_weight:bool = False,
                       is_train:bool = False):
    """Predict target labels based on majority voting from neighbours

    Args:
        distance_matrix (np.ndarray): a matrix with dimension n_train x n_target
        y_train (np.ndarray): a list of labels with the length of n_train
        n_neighbor (int): number of neighbours (on each side) used for prediction
        positive_weight_ratio (float, optional): Weight of positive labels relative to negative labels. Defaults to 1.0.
        direction (str, optional): One of "both", "close", or "far".
            When close/far, only closest/furthest (and reversed effect) neighbours will be used for prediction. Defaults to 'both'.
        flag_distance_weight (bool, optional): Whether to consider neighbour distance as a weight for majority voting.
            For close neighbours, closer neighbours have higher positive weights, and vice versa for far neighbours. Defaults to False.
        is_train (bool, optional): Flag to indicate whether this is the training set.
            If so, will remove itself from the matrix. Defaults to False.

    Returns:
        np.ndarray: prediction list
        np.ndarray: prediction probability list
    """
    if (not is_train):
        top_train_idx = np.argsort(distance_matrix, axis = 0)[:n_neighbor,:].T # closest, y_test x n_neighbor
        bottom_train_idx = np.argsort(distance_matrix, axis = 0)[-n_neighbor:,:].T # furthest, , y_test x n_neighbor
    else:
        top_train_idx = np.argsort(distance_matrix, axis = 0)[1:(n_neighbor+1),:].T # excluding itself
        bottom_train_idx = np.argsort(distance_matrix, axis = 0)[-(n_neighbor+1):-1,:].T

    top_train_label = y_train[top_train_idx]
    bottom_train_label = 1 - y_train[bottom_train_idx] # reverse prediction
    
    if (not flag_distance_weight): # weights equal 1
        if (direction == "both"):
            sum_label = np.sum(top_train_label + bottom_train_label, axis = 1)
            y_pred = (positive_weight_ratio * sum_label) > n_neighbor
            y_pred_prob = sum_label / (2 * n_neighbor)
        elif (direction == "close"):
            sum_label = np.sum(top_train_label, axis = 1)
            y_pred = (positive_weight_ratio * sum_label) > (n_neighbor / 2)
            y_pred_prob = sum_label / n_neighbor
        elif (direction == "far"):
            sum_label = np.sum(bottom_train_label, axis = 1)
            y_pred = (positive_weight_ratio * sum_label) > (n_neighbor / 2)
            y_pred_prob = sum_label / n_neighbor
        y_pred_prob = np.stack([1-y_pred_prob, y_pred_prob]).T
    else:
        top_train_label_weights = np.stack([distance_matrix[i_s, j] for i_s, j in zip(top_train_idx, range(len(top_train_idx)))])
        top_train_label_weights = 2 ** (np.abs(1 - top_train_label_weights)) - 1
        bottom_train_label_weights = np.stack([distance_matrix[i_s, j] for i_s, j in zip(bottom_train_idx, range(len(top_train_idx)))])
        bottom_train_label_weights = 2 ** (np.abs(1 - bottom_train_label_weights)) - 1
        if (direction == "both"):
            sum_label = np.sum(top_train_label * top_train_label_weights + bottom_train_label * bottom_train_label_weights, axis = 1)
            weight_sum = np.sum(top_train_label_weights+bottom_train_label_weights, axis = 1)
        elif (direction == "close"):
            sum_label = np.sum(top_train_label * top_train_label_weights, axis = 1)
            weight_sum = np.sum(top_train_label_weights, axis = 1)
        elif (direction == "far"):
            sum_label = np.sum(bottom_train_label * bottom_train_label_weights, axis = 1)
            weight_sum = np.sum(bottom_train_label_weights, axis = 1)
        y_pred = (positive_weight_ratio * sum_label) > (weight_sum / 2)
        y_pred_prob = sum_label / weight_sum
        y_pred_prob = np.stack([1-y_pred_prob, y_pred_prob]).T
    return y_pred, y_pred_prob
