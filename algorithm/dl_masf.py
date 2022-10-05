"""
Implementation of Model-Agnostic Learning of Semantic Features (MASF) algorithm for depression detection

reference:

Qi Dou, Daniel C. Castro, Konstantinos Kamnitsas, and Ben Glocker. 2019.
Domain Generalization via Model-Agnostic Learning of Semantic Features.
arXiv:1910.13580 [cs] (Oct. 2019). http://arxiv.org/abs/1910.13580 arXiv: 1910.13580
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *

from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm, DepressionDetectionClassifier_DL_erm
from data_loader.data_loader_ml import DataRepo
from utils import network, tf_metric_loss_64bit
import warnings


class DepressionDetectionClassifier_DL_masf(DepressionDetectionClassifier_DL_erm):
    """ MASF network classifier, extended from ERM classifier """

    def __init__(self, config):
        super().__init__(config=config)

        self.clf = self.clf_model(self.model_params)
        self.metric_network = Sequential()
        self.metric_network.add(Input(shape=(self.model_params["embedding_size"])))
        for shape in self.model_params["metric_network_fc_shapes"]:
            self.metric_network.add(Dense(shape, activation=keras.layers.LeakyReLU(alpha=0.2),
                kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform'))
        self.metric_network.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))

        if (self.model_params["num_domain"] == 1):
            warnings.warn("Only have one domain. Degrade to local loss only")

    class clf_model(Model):
        def __init__(self, model_params):
            super().__init__()
            self.model_params = model_params
            if (self.model_params["arch"] == "1dCNN"):
                self.feature_extractor = network.build_1dCNN(**model_params)
            elif (self.model_params["arch"] == "2dCNN"):
                self.feature_extractor = network.build_2dCNN(**model_params)
            elif (self.model_params["arch"] == "LSTM"):
                self.feature_extractor = network.build_LSTM(**model_params)
            elif (self.model_params["arch"] == "Transformer"):
                self.feature_extractor = network.build_Transformer(**model_params)

            self.label_predictor_layer = Dense(2)
            self.activation_layer = layers.Activation(activations.softmax)
        def call(self, x, is_training = True, return_logits=False):
            if (is_training):
                feature = self.feature_extractor(x)
            else:
                feature = self.feature_extractor.predict(x)
            label_logits = self.label_predictor_layer(feature)
            label_prob = self.activation_layer(label_logits)
            if return_logits:
                return label_logits
            else:
                return label_prob
    
    @tf.function
    def kl_divergence_symm(self, data1, label1, data2, label2):    
        kd_loss = 0.0
        eps = 1e-16
        temperature = self.model_params["kl_divergence_temperature"]
        n_class = 2

        for cls in range(n_class):
            mask1 = tf.tile(tf.expand_dims(label1[:, cls], -1), [1, n_class])
            logits_sum1 = tf.reduce_sum(input_tensor=tf.multiply(data1, mask1), axis=0)
            num1 = tf.reduce_sum(input_tensor=label1[:, cls])
            activations1 = logits_sum1 / (num1 + eps) # add eps for prevent un-sampled class resulting in NAN
            prob1 = tf.nn.softmax(activations1 / temperature)
            prob1 = tf.clip_by_value(prob1, clip_value_min=1e-8, clip_value_max=1.0)  # for preventing prob=0 resulting in NAN

            mask2 = tf.tile(tf.expand_dims(label2[:, cls], -1), [1, n_class])
            logits_sum2 = tf.reduce_sum(input_tensor=tf.multiply(data2, mask2), axis=0)
            num2 = tf.reduce_sum(input_tensor=label2[:, cls])
            activations2 = logits_sum2 / (num2 + eps)
            prob2 = tf.nn.softmax(activations2 / temperature)
            prob2 = tf.clip_by_value(prob2, clip_value_min=1e-8, clip_value_max=1.0)

            KL_div = (tf.reduce_sum(input_tensor=prob1 * tf.math.log(prob1 / prob2)) +\
                tf.reduce_sum(input_tensor=prob2 * tf.math.log(prob2 / prob1))) / 2.0
            kd_loss += KL_div

        kd_loss = kd_loss / n_class

        return kd_loss
                
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
        training_params_metric_network = deepcopy(self.training_params)
        training_params_metric_network['optimizer'] = training_params_metric_network['metric_network_optimizer']
        training_params_metric_network['learning_rate'] = training_params_metric_network['metric_network_learning_rate']
        metric_network_model_optimizer = network.prep_model_optimizer(training_params_outer)

        self.prep_callbacks(X)

        for data, label in X["val_whole"] if self.flag_X_dict else X: break
        self.clf(data) # init weights
        num_meta_train = int(self.model_params["num_domain"] * self.model_params["meta_training_proportion"])
        num_meta_test = self.model_params["num_domain"] - num_meta_train
        if (self.training_params["verbose"] > 1):
            print("matrix calc size", num_meta_train, num_meta_test)

        if (self.training_params.get("skip_training", False) == False):

            train_epoch_counter = 0
            pbar = tqdm(total = self.training_params["epochs"])
            train_step_counter = 0
            train_loss_record = []
            log_history = []
            data_batch_list = []

            for data, label in X["train"] if self.flag_X_dict else X:
                data_batch_list.append(data)
                if (len(data_batch_list) == self.model_params["num_domain"]):
                    data_batch_list = np.array(data_batch_list, dtype = object)

                    if (self.model_params["num_domain"] > 1):
                        task_list = np.random.permutation(self.model_params["num_domain"])
                        meta_train_index_list = task_list[:num_meta_train]
                        meta_test_index_list = task_list[num_meta_train:]

                        # skip when there is single y class
                        if ((sum(np.argmax(np.concatenate([data_batch_list[idx]["input_y"] for idx in meta_train_index_list]), axis = 1)) == num_meta_train) or
                            (sum(np.argmax(np.concatenate([data_batch_list[idx]["input_y"] for idx in meta_test_index_list]), axis = 1)) == num_meta_test)
                        ):
                            data_batch_list = []
                            continue

                        input_metatrain = [data_batch_list[idx] for idx in meta_train_index_list]
                        input_metatest = [data_batch_list[idx] for idx in meta_test_index_list]
                        input_concat = {}
                        for k in data_batch_list[0]:
                            input_concat[k] = tf.concat([data[k] for data in data_batch_list], axis = 0)
                        
                        old_vars = self.clf.get_weights()
                        with tf.GradientTape(persistent=True) as tape1:
                            task_loss_raw_train_list = []
                            task_loss_train_list = []
                            task_pred_list = []
                            for data in input_metatrain:
                                pred = self.clf(data)
                                task_pred_list.append(pred)
                                loss = keras.metrics.categorical_crossentropy(y_pred=pred, y_true=data["input_y"])
                                task_loss_raw_train_list.append(loss)
                                task_loss_train_list.append(tf.reduce_mean(loss))
                            task_loss_train = tf.concat(task_loss_raw_train_list, axis = 0)
                            task_loss_train_mean = tf.reduce_mean(task_loss_train_list)
                        grads = tape1.gradient(target=task_loss_train, sources=self.clf.trainable_variables)

                        # After the meta-learning step, reload the newly-trained weights into the model.
                        grads = [tf.stop_gradient(grad) for grad in grads]
                        grads = [tf.clip_by_norm(grad, clip_norm=self.training_params["gradient_clip_norm"]) for grad in grads]
                        inner_model_optimizer.apply_gradients(zip(grads, self.clf.trainable_variables))
                        
                        with tf.GradientTape(persistent=True) as tape2:
                            # global loss
                            logit_train_list = []
                            for data in input_metatrain:
                                pred = self.clf(data, return_logits=True)
                                logit_train_list.append(pred)
                            logit_test_list = []
                            for data in input_metatest:
                                pred = self.clf(data, return_logits=True)
                                logit_test_list.append(pred)
                            global_loss_list = []
                            for logit_train, data_train in zip(logit_train_list, input_metatrain):
                                for logit_test, data_test in zip(logit_test_list, input_metatest):
                                    global_loss_list.append(self.kl_divergence_symm(
                                                            data1=logit_test, label1=data_test["input_y"],
                                                            data2=logit_train, label2=data_train["input_y"]))
                            global_loss = tf.reduce_mean(global_loss_list)
                        
                            # local loss
                            features = self.clf.feature_extractor(input_concat)
                            embeddings = self.metric_network(features)
                            local_loss = tf_metric_loss_64bit.triplet_semihard_loss_64bit(y_true=tf.argmax(input_concat["input_y"], axis=1),
                                                                        y_pred=embeddings, margin=self.model_params["triplet_loss_margin"])
                            global_local_loss = global_loss * self.model_params["global_loss_weight"] +\
                                local_loss * self.model_params["local_loss_weight"]

                    else: # only have ERM + local loss
                        task_list = np.random.permutation(self.model_params["num_domain"])
                        meta_train_index_list = task_list

                        # skip when there is single y class
                        if ((sum(np.argmax(np.concatenate([data_batch_list[idx]["input_y"] for idx in meta_train_index_list]), axis = 1)) == num_meta_train)):
                            data_batch_list = []
                            continue

                        input_metatrain = [data_batch_list[idx] for idx in meta_train_index_list]
                        input_concat = {}
                        for k in data_batch_list[0]:
                            input_concat[k] = tf.concat([data[k] for data in data_batch_list], axis = 0)
                        
                        old_vars = self.clf.get_weights()
                        with tf.GradientTape(persistent=True) as tape1:
                            task_loss_raw_train_list = []
                            task_loss_train_list = []
                            task_pred_list = []
                            for data in input_metatrain:
                                pred = self.clf(data)
                                task_pred_list.append(pred)
                                loss = keras.metrics.categorical_crossentropy(y_pred=pred, y_true=data["input_y"])
                                task_loss_raw_train_list.append(loss)
                                task_loss_train_list.append(tf.reduce_mean(loss))
                            task_loss_train = tf.concat(task_loss_raw_train_list, axis = 0)
                            task_loss_train_mean = tf.reduce_mean(task_loss_train_list)
                        grads = tape1.gradient(target=task_loss_train, sources=self.clf.trainable_variables)

                        # After the meta-learning step, reload the newly-trained weights into the model.
                        grads = [tf.stop_gradient(grad) for grad in grads]
                        grads = [tf.clip_by_norm(grad, clip_norm=self.training_params["gradient_clip_norm"]) for grad in grads]
                        inner_model_optimizer.apply_gradients(zip(grads, self.clf.trainable_variables))
                        
                        with tf.GradientTape(persistent=True) as tape2:
                            # global loss
                            global_loss = 0
                        
                            # local loss
                            features = self.clf.feature_extractor(input_concat)
                            embeddings = self.metric_network(features)
                            local_loss = tf_metric_loss_64bit.triplet_semihard_loss_64bit(y_true=tf.argmax(input_concat["input_y"], axis=1),
                                                                        y_pred=embeddings, margin=self.model_params["triplet_loss_margin"])
                            global_local_loss = local_loss * self.model_params["local_loss_weight"]
                        
                    # go back to old vars and train outer
                    train_loss_record.append([task_loss_train_mean+global_local_loss, task_loss_train_mean, global_loss, local_loss])

                    self.clf.set_weights(old_vars)
                    grads_source = tape1.gradient(target=task_loss_train_mean, sources=self.clf.trainable_variables)
                    outer_model_optimizer.apply_gradients(zip(grads_source, self.clf.trainable_variables))
                    
                    grads_globallocal = tape2.gradient(target = global_local_loss, sources = self.clf.trainable_variables)
                    grads_globallocal = [tf.stop_gradient(grad) if grad is not None else None for grad in grads_globallocal]
                    grads_globallocal = [tf.clip_by_norm(grad, clip_norm=self.training_params["gradient_clip_norm"]) if grad is not None else None for grad in grads_globallocal]
                    outer_model_optimizer.apply_gradients(
                        (grad, var) 
                        for (grad, var) in zip(grads_globallocal, self.clf.trainable_variables) 
                        if grad is not None
                    )
                    
                    grads_metric = tape2.gradient(target=local_loss, sources = self.metric_network.trainable_variables)
                    metric_network_model_optimizer.apply_gradients(zip(grads_metric, self.metric_network.trainable_variables))
                    
                    del tape1
                    del tape2

                    train_step_counter += 1
                    data_batch_list = []

                    if (train_step_counter % self.training_params["steps_per_epoch"] == 0):
                        self.__assert__(X) # ensure the flag_X_dict is correct
                        train_epoch_counter += 1
                        pbar.update(1)
                        train_loss, train_task_loss, train_global_loss, train_local_loss = tf.reduce_mean(train_loss_record, axis = 0).numpy()
                        # calc val loss
                        for data_val, label_val in (X["val"] if self.flag_X_dict else X):
                            pred_val = self.clf.predict(data_val)
                            loss_val = tf.reduce_mean(
                                tf.keras.metrics.categorical_crossentropy(y_pred=pred_val, y_true=data_val["input_y"]))

                        logs = {"loss": train_loss, "loss_task": train_task_loss,
                            "loss_global": train_global_loss, "loss_local": train_local_loss, "val_loss":loss_val.numpy()}
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


class DepressionDetectionAlgorithm_DL_masf(DepressionDetectionAlgorithm_DL_erm):
    """ The MASF algorithm. Extends the ERM algorithm """

    def __init__(self, config_dict = None, config_name = "dl_masf"):
        super().__init__(config_dict, config_name)
        assert self.config["data_loader"]["generate_by"] in ["within_person", "within_dataset"]
        
    
    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        self.config["model_params"].update(
            {"input_shape": self.input_shape,
            "flag_return_embedding":True, "flag_embedding_norm":False,
            "flag_input_dict":True}
        )
        if (self.config["data_loader"]["generate_by"] == "within_person"):
            num_domain = len(self.data_generator_whole.person_dict)
        else:
            num_domain = len(self.data_generator_whole.dataset_dict)
        self.config["model_params"]["num_domain"] = num_domain
        
        return DepressionDetectionClassifier_DL_masf(config=self.config)