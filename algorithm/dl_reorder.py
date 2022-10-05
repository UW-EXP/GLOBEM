"""
Implementation of a reordering-based multi-task learning algorithm for depression detection

"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *
from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm, DepressionDetectionClassifier_DL_erm
from data_loader.data_loader_dl import MultiSourceDataGenerator
from data_loader.data_loader_ml import DataRepo
from utils import network, path_definitions


class EvaluationCallback_reorder(network.EvaluationBasicCallback):
    """Evaluation callback function for reoder model"""
    def __init__(self, model_obj, dataset_train, dataset_test, interval=1, verbose=1):
        super().__init__(model_obj, dataset_train, dataset_test, interval, verbose, flag_skip_y_defition=True)

        # redefine y while ignoring the reorder label
        self.y_train = np.array([i[1][0] for i in self.dataset_train][0])
        if (len(self.y_train.shape) > 1): # if y is a vector, convert to sparse
            self.y_train = np.argmax(self.y_train, axis = 1)
        if (self.flag_with_test):
            self.y_test = np.array([i[1][0] for i in self.dataset_test][0])
            if (len(self.y_test.shape) > 1): # if y is a vector, convert to sparse
                self.y_test = np.argmax(self.y_test, axis = 1)

    def on_epoch_end(self, epoch, logs=None):
        # to be overwritten
        if (epoch % self.interval == 0):
            results_train = utils_ml.results_report_sklearn(clf = self.model_obj,
                X=self.dataset_train, y=self.y_train, return_confusion_mtx=True)
            if (self.flag_with_test):
                results_test = utils_ml.results_report_sklearn(clf = self.model_obj,
                    X=self.dataset_test, y=self.y_test, return_confusion_mtx=True)
            else:
                results_test = None
            new_logs = {}
            for k, v in logs.items(): # rename the built in log keys to make it consistent with other models
                if ("output_1_" in k):
                    k_new = k.replace("output_1_", "")
                else:
                    k_new = k
                new_logs[k_new] = v
            self.process_results(epoch, new_logs, results_train, results_test)

class DepressionDetectionClassifier_DL_reorder(DepressionDetectionClassifier_DL_erm):
    """ Reorder classifier, extended from ERM classifier """

    def __init__(self, config):
        super().__init__(config=config)

        self.clf = self.reorder_model(self.model_params)

    class reorder_model(Model):
        def __init__(self, model_params):
            super().__init__()
            #Feature Extractor
            self.model_params = model_params

            if (self.model_params["arch"] == "1dCNN"):
                self.feature_extractor = network.build_1dCNN(**model_params)
            elif (self.model_params["arch"] == "2dCNN"):
                self.feature_extractor = network.build_2dCNN(**model_params)
            elif (self.model_params["arch"] == "LSTM"):
                self.feature_extractor = network.build_LSTM(**model_params)
            elif (self.model_params["arch"] == "Transformer"):
                self.feature_extractor = network.build_Transformer(**model_params)
            
            #Label Predictor
            self.label_predictor_layer0 = Dense(16, activation='relu')
            self.label_predictor_layer1 = Dense(2, activation="softmax")
            
            #Domain Predictor
            self.domain_predictor_layer0 = Dense(32, activation='relu')
            self.domain_predictor_layer1 = Dense(model_params["num_reorder_class"] + 1, activation="softmax")
            
        def call(self, x, is_training = True):
            if (is_training):
                feature = self.feature_extractor(x)
            else:
                feature = self.feature_extractor.predict(x)
            
            lp_x = self.label_predictor_layer0(feature)
            label_prob = self.label_predictor_layer1(lp_x)

            dp_x = self.domain_predictor_layer0(feature)
            reorder_prob = self.domain_predictor_layer1(dp_x)

            return label_prob, reorder_prob

    def prep_eval_callbacks(self, X):
        ds_train = X["val_whole"]
        if "test" in X:
            ds_test = X["test"]
        elif "val" in X:
            ds_test = X["val"]
        else:
            ds_test = None
        return EvaluationCallback_reorder(model_obj=self,
                dataset_train=ds_train, dataset_test=ds_test,
                interval=1, verbose=self.training_params["verbose"])

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(42)

        self.__assert__(X)

        model_optimizer = network.prep_model_optimizer(self.training_params)

        self.clf.compile(loss = ['categorical_crossentropy','categorical_crossentropy'],
            loss_weights = [1, self.model_params["weight_of_reorder"]], metrics="acc",
            optimizer = model_optimizer)

        callbacks = self.prep_callbacks(X)

        if (self.training_params.get("skip_training", False) == False):

            history = self.clf.fit(x = X["train"] if self.flag_X_dict else X,
                    steps_per_epoch = self.training_params["steps_per_epoch"],
                    epochs = self.training_params["epochs"],
                    validation_data = X["val"] if self.flag_X_dict else X,
                    verbose = 1 if self.training_params["verbose"] > 1 else 0,
                    callbacks = callbacks
                    )

            self.log_history = history.history
            
            best_epoch, df_results_record = self.find_best_epoch()

            self.clf.set_weights(self.model_saver.model_repo_dict[best_epoch])
        else:
            # fit one step to initialize all layers
            history = self.clf.fit(x = X["train"] if self.flag_X_dict else X,
                    steps_per_epoch = 1, epochs = 1, verbose = 0)

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

class MultiSourceDataGeneratorReorder(MultiSourceDataGenerator):
    def __init__(self, 
                data_repo_dict: Dict[str, DataRepo], is_training = True,
                generate_by = "across_dataset", 
                batch_size=32, shuffle=True, flag_y_vector=True,
                mixup = "across", mixup_alpha=0.2,
                **kwargs,
                ):
        super().__init__(data_repo_dict=data_repo_dict,
            is_training=is_training,
            generate_by=generate_by,
            batch_size=batch_size,
            shuffle=shuffle,
            flag_y_vector=flag_y_vector,
            mixup = mixup, mixup_alpha = mixup_alpha)
        self.num_reorder_classes = kwargs.get("num_reorder_classes", 100)
        self.rate_of_reorder = kwargs.get("rate_of_reorder", 0.5)
        self.permutation_list_raw = np.load(
            os.path.join(path_definitions.TMP_PATH, f'reorder/permutations_hamming_max_{self.num_reorder_classes}.npy'))
        self.noshuffle_idx = np.arange(28)
        self.permutation_list = []
        for p in self.permutation_list_raw:
            d = []
            for idx in p:
                if (idx == 9):
                    d += [idx * 3]
                else:
                    d += [idx * 3, idx*3 + 1, idx*3 + 2]
            self.permutation_list.append(np.array(d))
        self.permutation_list = np.array(self.permutation_list)

        self.tf_output_signature = ({
                    "input_X": tf.TensorSpec(shape=[None] + self.input_shape, dtype = tf.float64),
                    "input_y": tf.TensorSpec(shape=(None, 2) if self.flag_y_vector else (None), dtype = tf.float64),
                    "input_dataset": tf.TensorSpec(shape=(None), dtype = tf.int64),
                    "input_person": tf.TensorSpec(shape=(None), dtype = tf.int64),
                },
            (tf.TensorSpec(shape=(None, 2) if self.flag_y_vector else (None), dtype = tf.float64),
            tf.TensorSpec(shape=(None, self.num_reorder_classes + 1), dtype = tf.float64))
        )

    def __call__(self):
        generator = super().__call__()
        for data, label in generator:
            batch_size = len(label)
            batch_size_shuffle = int(batch_size * self.rate_of_reorder)
            reorder_labels_shuffle = np.random.randint(low = 1, high = self.num_reorder_classes + 1, size = batch_size_shuffle)
            reorder_labels = np.concatenate([[0 for _ in range(batch_size - batch_size_shuffle)], reorder_labels_shuffle])
            reorder_labels = tf.keras.utils.to_categorical(reorder_labels, num_classes = self.num_reorder_classes + 1)
            
            reorder_idx = np.concatenate([[self.noshuffle_idx for _ in range(batch_size - batch_size_shuffle)],
                self.permutation_list[reorder_labels_shuffle-1]])
            
            data["input_X"] = np.array([x[idx,:] for x, idx in zip(data["input_X"],reorder_idx)])
            yield data, (label, reorder_labels)

class DepressionDetectionAlgorithm_DL_reorder(DepressionDetectionAlgorithm_DL_erm):
    """ The Reorder algorithm. Extends the ERM algorithm """

    def __init__(self, config_dict = None, config_name = "dl_reorder"):
        super().__init__(config_dict, config_name)
        self.data_generator_obj = MultiSourceDataGeneratorReorder
        self.data_generator_additional_args = {"train":{"num_reorder_classes": self.config["model_params"]["num_reorder_class"],
                                               "rate_of_reorder": self.config["model_params"]["rate_of_reorder"]},
                                               "nontrain":{"num_reorder_classes": self.config["model_params"]["num_reorder_class"],
                                               "rate_of_reorder": 0}}
        
    
    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        self.config["model_params"].update(
            {"input_shape": self.input_shape,
            "flag_return_embedding":True, "flag_embedding_norm":False,
            "flag_input_dict":True}
        )
        return DepressionDetectionClassifier_DL_reorder(config = self.config)