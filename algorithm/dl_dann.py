"""
Implementation of Domain-Adversarial Neural Network (DANN) algorithm for depression detection

reference:

Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle,
François Laviolette, Mario Marchand, and Victor Lempitsky. 2017. 
Domain-Adversarial Training of Neural Networks. In Domain Adaptation in Computer Vision Applications.
Springer International Publishing, Cham, 189–209. https://doi.org/10.1007/978-3-319-58347-1_10
Series Title: Advances in Computer Vision and Pattern Recognition.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm, DepressionDetectionClassifier_DL_erm
from data_loader.data_loader_ml import DataRepo
from utils import network

@tf.custom_gradient
def gradient_reverse(x, lamda=1.0):
    y = tf.identity(x)
    
    def grad(dy):
        return lamda * -dy, None
    
    return y, grad

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x, lamda=1.0):
        return gradient_reverse(x, lamda)

class DepressionDetectionClassifier_DL_dann(DepressionDetectionClassifier_DL_erm):
    """ DANN network classifier, extended from ERM classifier """

    def __init__(self, config):
        super().__init__(config=config)
        self.clf = self.dann_model(model_params=self.model_params)

    class dann_model(Model):
        def __init__(self, model_params):
            super().__init__()
            self.model_params = model_params
            #Feature Extractor
            if (self.model_params["arch"] == "1dCNN"):
                self.feature_extractor = network.build_1dCNN(**model_params)
            elif (self.model_params["arch"] == "2dCNN"):
                self.feature_extractor = network.build_2dCNN(**model_params)
            elif (self.model_params["arch"] == "LSTM"):
                self.feature_extractor = network.build_LSTM(**model_params)
            elif (self.model_params["arch"] == "Transformer"):
                self.feature_extractor = network.build_Transformer(**model_params)
            
            #Label Predictor
            self.label_predictor_layer1 = Dense(2, activation="softmax")
            
            #Domain Predictor
            self.domain_predictor_layer0 = GradientReversalLayer()
            self.domain_predictor_layer1 = Dense(32, activation='relu')
            self.domain_predictor_layer2 = Dense(model_params["num_domain"], activation="softmax")
            
        def call(self, x, is_training = True, reverse_gradient_weights=1.0):
            #Feature Extractor
            if (is_training):
                feature = self.feature_extractor(x)
            else:
                feature = self.feature_extractor.predict(x)
            
            l_prob = self.label_predictor_layer1(feature)

            dp_x = self.domain_predictor_layer0(feature, reverse_gradient_weights) #GradientReversalLayer
            dp_x = self.domain_predictor_layer1(dp_x)
            d_prob = self.domain_predictor_layer2(dp_x)

            return l_prob, d_prob

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(42)

        self.__assert__(X)

        model_optimizer = network.prep_model_optimizer(self.training_params)

        self.prep_callbacks(X)
        
        if (self.model_params["target_domain"] == "person"):
            domain_key = "input_person"
        elif (self.model_params["target_domain"] == "dataset"):
            domain_key = "input_dataset"

        if (self.training_params.get("skip_training", False) == False):
            pbar = tqdm(total = self.training_params["epochs"])
            train_epoch_counter = 0
            train_step_counter = 0
            train_loss_record = []
            log_history = []

            for data, label in X["train"] if self.flag_X_dict else X:
                
                reverse_gradient_weights = (self.model_params["reverse_gradient_init_weights"] + 1) / \
                    (1 + np.exp(-100.0 * train_epoch_counter / self.training_params['epochs'], dtype=np.float64)) - 1
                reverse_gradient_weights = tf.cast(reverse_gradient_weights, tf.float64)

                with tf.GradientTape() as tape:
                    label_prob, domain_prob = self.clf(data, is_training = True,
                        reverse_gradient_weights=reverse_gradient_weights)
                    
                    loss_label = tf.reduce_mean(keras.metrics.categorical_crossentropy(y_pred=label_prob,
                                    y_true=data["input_y"]))
                    loss_domain = tf.reduce_mean(keras.metrics.categorical_crossentropy(y_pred=domain_prob,
                                    y_true=keras.utils.to_categorical(data[domain_key], num_classes = self.model_params["num_domain"])))
                    model_loss = loss_label + loss_domain
                train_loss_record.append([loss_label, loss_domain, model_loss])
                    
                gradients = tape.gradient(model_loss, self.clf.trainable_variables)
                model_optimizer.apply_gradients(zip(gradients, self.clf.trainable_variables))

                train_step_counter += 1
                if (train_step_counter % self.training_params["steps_per_epoch"] == 0):
                    self.__assert__(X) # ensure the flag_X_dict is correct
                    train_epoch_counter += 1
                    pbar.update(1)
                    train_loss, train_loss_domain, train_loss_sum = tf.reduce_mean(train_loss_record, axis = 0).numpy()
                    # calc val loss
                    for data_val, label_val in (X["val"] if self.flag_X_dict else X):
                        pred_prob, domain_prob = self.clf.predict(data_val)
                        loss_label_val = tf.reduce_mean(keras.metrics.categorical_crossentropy(y_pred=pred_prob,
                            y_true=data_val["input_y"]))
                        loss_domain_val = tf.reduce_mean(keras.metrics.categorical_crossentropy(y_pred=domain_prob,
                            y_true=keras.utils.to_categorical(data_val[domain_key], num_classes = self.model_params["num_domain"])))

                    logs = {"loss": train_loss, "loss_domain": train_loss_domain, "loss_sum": train_loss_sum,
                        "val_loss":loss_label_val.numpy(), "val_loss_domain": loss_domain_val.numpy()}
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
            # take one step to initialize
            for data, label in X["train"] if self.flag_X_dict else X:
                reverse_gradient_weights = tf.cast(1, tf.float64)
                with tf.GradientTape() as tape:
                    label_prob, domain_prob = self.clf(data, is_training = True,
                        reverse_gradient_weights=reverse_gradient_weights)
                    loss_label = tf.reduce_mean(keras.metrics.categorical_crossentropy(y_pred=label_prob,
                                    y_true=data["input_y"]))
                    loss_domain = tf.reduce_mean(keras.metrics.categorical_crossentropy(y_pred=domain_prob,
                                    y_true=keras.utils.to_categorical(data[domain_key], num_classes = self.model_params["num_domain"])))
                    model_loss = loss_label + loss_domain
                gradients = tape.gradient(model_loss, self.clf.trainable_variables)
                model_optimizer.apply_gradients(zip(gradients, self.clf.trainable_variables))
                break

            df_results_record = self.fit_skip_training()

        return df_results_record

    def predict(self, X, y=None):
        self.__assert__(X)
        if (self.flag_X_dict):
            X_ = X["val_whole"] # only use the whole val set for eval
        else:
            X_ = X
        for data, label in X_:
            return np.argmax(self.clf.predict(data)[0], axis = 1)

    def predict_proba(self, X, y=None):
        self.__assert__(X)
        if (self.flag_X_dict):
            X_ = X["val_whole"] # only use the whole val set for eval
        else:
            X_ = X
        for data, label in X_:
            return self.clf.predict(data)[0]


class DepressionDetectionAlgorithm_DL_dann(DepressionDetectionAlgorithm_DL_erm):
    """ The DANN algorithm. Extends the ERM algorithm """

    def __init__(self, config_dict = None, config_name = "dl_dann"):
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
            self.config["model_params"]["num_domain"] = len(self.data_generator_whole.person_dict)
            self.config["model_params"]["target_domain"] = "person"
        else:
            self.config["model_params"]["num_domain"] = len(self.data_generator_whole.dataset_dict)
            self.config["model_params"]["target_domain"] = "dataset"

        self.config["training_params"]["step_batch_size"] = self.data_generator_whole.step_size

        return DepressionDetectionClassifier_DL_dann(config=self.config)