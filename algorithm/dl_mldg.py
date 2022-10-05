"""
Implementation of Meta-Learning for Domain Generalization (MLDG) algorithm for depression detection

reference:

Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M. Hospedales. 2017.
Learning to Generalize: Meta-Learning for Domain Generalization.
arXiv:1710.03463 [cs] (Oct. 2017). http://arxiv.org/abs/1710.03463 arXiv: 1710.03463.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm, DepressionDetectionClassifier_DL_erm
from data_loader.data_loader_ml import DataRepo
from utils import network
import warnings

class DepressionDetectionClassifier_DL_mldg(DepressionDetectionClassifier_DL_erm):
    """ MLDG network classifier, extended from ERM classifier """
    
    def __init__(self, config):
        super().__init__(config = config)

        if (self.model_params["num_domain"] == 1):
            warnings.warn("Only have one domain. Degrade to basic ERM")

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(42)

        self.__assert__(X)

        training_params_inner = deepcopy(self.training_params)
        training_params_inner['optimizer'] = training_params_inner['inner_optimizer']
        training_params_inner['learning_rate'] = training_params_inner['inner_learning_rate']
        inner_model_optimizer = network.prep_model_optimizer(training_params_inner)
        training_params_outer = deepcopy(self.training_params)
        training_params_outer['optimizer'] = training_params_outer['outer_optimizer']
        training_params_outer['learning_rate'] = training_params_outer['outer_learning_rate']
        outer_model_optimizer = network.prep_model_optimizer(training_params_outer)

        self.prep_callbacks(X)
        
        num_meta_train = int(self.model_params["num_domain"] * self.model_params["meta_training_proportion"])
        batch_size = self.config["data_loader"]["step_batch_size"]

        if (self.training_params.get("skip_training", False) == False):

            train_epoch_counter = 0
            pbar = tqdm(total = self.training_params["epochs"])
            train_step_counter = 0
            train_loss_record = []
            log_history = []

            for data, label in X["train"] if self.flag_X_dict else X:
                task_list = np.random.permutation(self.model_params["num_domain"])
                if (self.model_params["num_domain"] > 1):
                    meta_train_index_list_idx = task_list[:num_meta_train]
                    meta_test_index_list_idx = task_list[num_meta_train:]
                    meta_train_index_list = np.concatenate(
                        [np.arange(idx * batch_size, (idx + 1) * batch_size) for idx in meta_train_index_list_idx])
                    meta_test_index_list = np.concatenate(
                        [np.arange(idx * batch_size, (idx + 1) * batch_size) for idx in meta_test_index_list_idx])
                    data_metatrain = {}
                    data_metatest = {}
                    for k in data:
                        data_metatrain[k] = tf.gather(data[k], meta_train_index_list, axis = 0)
                        data_metatest[k] = tf.gather(data[k], meta_test_index_list, axis = 0)

                    old_vars = self.clf.get_weights()
                    with tf.GradientTape(persistent=True) as tape1:
                        pred_metatrain = self.clf(data_metatrain)
                        loss_metatrain = tf.reduce_mean(
                            tf.keras.metrics.categorical_crossentropy(y_pred=pred_metatrain, y_true=data_metatrain["input_y"]))
                    grads = tape1.gradient(target=loss_metatrain, sources=self.clf.trainable_variables)
                    inner_model_optimizer.apply_gradients(zip(grads, self.clf.trainable_variables))

                    with tf.GradientTape(persistent=False) as tape2:
                        pred_metatest = self.clf(data_metatest)
                        loss_metatest = tf.reduce_mean(
                            tf.keras.metrics.categorical_crossentropy(y_pred=pred_metatest, y_true=data_metatest["input_y"]))
                        loss_summary = loss_metatrain + self.model_params["meta_testing_weight"] * loss_metatest
                    train_loss_record.append([loss_summary, loss_metatrain, loss_metatest])

                    self.clf.set_weights(old_vars)
                    
                    grads_summary = tape2.gradient(target=loss_summary, sources=self.clf.trainable_variables)
                    outer_model_optimizer.apply_gradients(zip(grads_summary, self.clf.trainable_variables))
                    del tape1
                    del tape2
                else: # degrade to basic ERM
                    meta_train_index_list_idx = task_list
                    meta_train_index_list = np.concatenate(
                        [np.arange(idx * batch_size, (idx + 1) * batch_size) for idx in meta_train_index_list_idx])
                    data_metatrain = {}
                    for k in data:
                        data_metatrain[k] = tf.gather(data[k], meta_train_index_list, axis = 0)
                    old_vars = self.clf.get_weights()
                    with tf.GradientTape(persistent=False) as tape1:
                        pred_metatrain = self.clf(data_metatrain)
                        loss_metatrain = tf.reduce_mean(
                            tf.keras.metrics.categorical_crossentropy(y_pred=pred_metatrain, y_true=data_metatrain["input_y"]))
                    train_loss_record.append([loss_metatrain, loss_metatrain, 0])
                    grads_summary = tape1.gradient(target=loss_metatrain, sources=self.clf.trainable_variables)
                    outer_model_optimizer.apply_gradients(zip(grads_summary, self.clf.trainable_variables))

                train_step_counter += 1
                if (train_step_counter % self.training_params["steps_per_epoch"] == 0):
                    self.__assert__(X) # ensure the flag_X_dict is correct
                    train_epoch_counter += 1
                    pbar.update(1)
                    train_loss, train_loss_metatrain, train_loss_metatest = tf.reduce_mean(train_loss_record, axis = 0).numpy()
                    # calc val loss
                    for data_val, label_val in (X["val"] if self.flag_X_dict else X):
                        pred_val = self.clf.predict(data_val)
                        loss_val = tf.reduce_mean(
                            tf.keras.metrics.categorical_crossentropy(y_pred=pred_val, y_true=data_val["input_y"]))

                    logs = {"loss": train_loss, "loss_metatrain": train_loss_metatrain, "loss_metatest": train_loss_metatest, "val_loss":loss_val.numpy()}
                    if (self.training_params["verbose"] > 1):
                        print(f"Epoch: {train_epoch_counter}", end = " - ")
                        print(logs)

                    # manual callback
                    log_history.append(deepcopy(logs))
                    # save model
                    self.model_saver.model_repo_dict[train_epoch_counter] = deepcopy(self.clf.get_weights())
                    # calc metrics
                    self.eval_callback.on_epoch_end(epoch=train_epoch_counter, logs = logs)

                if (train_epoch_counter >= self.training_params["epochs"]):
                    pbar.close()
                    break
            self.log_history = log_history
            
            best_epoch, df_results_record = self.find_best_epoch()

            self.clf.set_weights(self.model_saver.model_repo_dict[best_epoch])
        else:
            df_results_record = self.fit_skip_training()


        return df_results_record


class DepressionDetectionAlgorithm_DL_mldg(DepressionDetectionAlgorithm_DL_erm):
    """ The MLDG algorithm. Extends the ERM algorithm """

    def __init__(self, config_dict = None, config_name = "dl_mldg"):
        super().__init__(config_dict, config_name)
        assert self.config["data_loader"]["generate_by"] in ["across_person", "across_dataset"]
        
    
    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        
        self.config["model_params"].update(
            {
            "input_shape": self.input_shape,
            "flag_return_embedding":False, "flag_embedding_norm":False,
            "flag_input_dict":True
            }
        )
        if (self.config["data_loader"]["generate_by"] == "across_person"):
            num_domain = len(self.data_generator_whole.person_dict)
        else:
            num_domain = len(self.data_generator_whole.dataset_dict)
        self.config["model_params"]["num_domain"] = num_domain

        self.config["data_loader"]["step_batch_size"] = self.data_generator_whole.step_size

        return DepressionDetectionClassifier_DL_mldg(config = self.config)