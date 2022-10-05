"""
Implementation of Invariant Risk Minimization (IRM) algorithm for depression detection

reference:

Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. 2020.
Invariant Risk Minimization. arXiv:1907.02893 [cs, stat]
(March 2020). http://arxiv.org/abs/1907.02893 arXiv: 1907.02893.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm, DepressionDetectionClassifier_DL_erm
from data_loader.data_loader_ml import DataRepo
from utils import network

class DepressionDetectionClassifier_DL_irm(DepressionDetectionClassifier_DL_erm):
    """ IRM network classifier, extended from ERM classifier """
    
    def __init__(self, config):
        super().__init__(config=config)

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(42)

        self.__assert__(X)

        model_optimizer = network.prep_model_optimizer(self.training_params)

        self.prep_callbacks(X)

        if (self.training_params.get("skip_training", False) == False):
            pbar = tqdm(total = self.training_params["epochs"])
            train_epoch_counter = 0
            train_step_counter = 0
            train_loss_record = []
            log_history = []

            for data, label in X["train"] if self.flag_X_dict else X:
                
                dummy = tf.convert_to_tensor([1.], dtype = tf.float64)
                with tf.GradientTape() as tape_irm:
                    tape_irm.watch(dummy)
                    dummy_logits = self.clf(data, training=True)
                    loss_irm = tf.reduce_mean(keras.metrics.categorical_crossentropy(y_true=data["input_y"], y_pred=dummy_logits * dummy))
                irm_grads = tape_irm.gradient(loss_irm, dummy)
                penalty_irm = tf.reduce_mean(irm_grads ** 2)

                with tf.GradientTape() as tape_erm:
                    label_prob = self.clf(data, training = True)
                    loss_erm = tf.reduce_mean(keras.metrics.categorical_crossentropy(y_true=data["input_y"], y_pred=label_prob))
                    irm_penalty_weights = self.irm_penalty_weights_schedule(train_epoch_counter)
                    model_loss = loss_erm + irm_penalty_weights * penalty_irm
                
                train_loss_record.append([model_loss, loss_erm, penalty_irm, loss_irm])
                gradients = tape_erm.gradient(model_loss, self.clf.trainable_variables)
                model_optimizer.apply_gradients(zip(gradients, self.clf.trainable_variables))

                train_step_counter += 1
                if (train_step_counter % self.training_params["steps_per_epoch"] == 0):
                    self.__assert__(X) # ensure the flag_X_dict is correct
                    train_epoch_counter += 1
                    pbar.update(1)
                    train_loss, train_loss_erm, train_penalty_irm, train_loss_irm = tf.reduce_mean(train_loss_record, axis = 0).numpy()
                    # calc val loss
                    for data_val, label_val in (X["val"] if self.flag_X_dict else X):
                        pred_prob = self.clf.predict(data_val)
                        loss_label_val = tf.reduce_mean(keras.metrics.categorical_crossentropy(y_pred=pred_prob,
                            y_true=data_val["input_y"]))

                    logs = {"loss": train_loss, "loss_erm": train_loss_erm, "penalty_irm": train_penalty_irm, "loss_irm": train_loss_irm,
                        "val_loss":loss_label_val.numpy()}
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

            self.log_history = pd.DataFrame(log_history)

            best_epoch, df_results_record = self.find_best_epoch()

            self.clf.set_weights(self.model_saver.model_repo_dict[best_epoch])
        else:
            df_results_record = self.fit_skip_training()

        return df_results_record

    def irm_penalty_weights_schedule(self, epoch):
        if epoch < 10:
            return 5 ** 13
        elif epoch < 20:
            return 3 * 10 **15
        else:
            return 10 ** 16 

class DepressionDetectionAlgorithm_DL_irm(DepressionDetectionAlgorithm_DL_erm):
    """ The IRM algorithm. Extends the ERM algorithm """

    def __init__(self, config_dict = None, config_name = "dl_irm"):
        super().__init__(config_dict, config_name)
        assert self.config["data_loader"]["generate_by"] in ["across_person", "across_dataset"]
        
    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        
        self.config["model_params"].update({
            "input_shape": self.input_shape,
            "flag_return_embedding":False, "flag_embedding_norm":False,
            "flag_input_dict":True
            })

        return DepressionDetectionClassifier_DL_irm(config = self.config)