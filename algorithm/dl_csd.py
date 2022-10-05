"""
Implementation of Common Specific Decomposition (CSD) algorithm for depression detection

reference:

Vihari Piratla, Praneeth Netrapalli, and Sunita Sarawagi. 2020.
Efficient Domain Generalization via Common-Specific Low-Rank Decomposition.
arXiv:2003.12815 [cs, stat] (April 2020). http://arxiv.org/abs/2003.12815 arXiv: 2003.12815.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *

from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm, DepressionDetectionClassifier_DL_erm
from data_loader.data_loader_ml import DataRepo
from utils import network

class DepressionDetectionClassifier_DL_csd(DepressionDetectionClassifier_DL_erm):
    """ CSD network classifier, extended from ERM classifier """

    def __init__(self, config):
        super().__init__(config=config)
        self.clf = self.build_csd_model()

    class csd_layer(layers.Layer):
        def __init__(self, embedding_size, common_specific_weights = 0.5, num_domain = 3, low_rank_dim = 20):
            super().__init__()
            self.num_classes = 2
            self.num_domain = num_domain
            self.common_specific_weights = common_specific_weights
            self.low_rank_dim = low_rank_dim
            self.embedding_size = embedding_size
            
            self.common_wt = tf.Variable(tf.ones([1]), name = "common_wt",  trainable=False)
            
            specialized_common_wt_init = tf.random_normal_initializer(.5, 1e-2)
            self.specialized_common_wt = tf.Variable(name = "specialized_common_wt",
                initial_value=specialized_common_wt_init(shape=[1], dtype=tf.float64),
                trainable=True)
            
            emb_matrix_init = tf.random_normal_initializer(0, 1e-4)
            self.emb_matrix = tf.Variable(name = "emb_matrix",
                initial_value=emb_matrix_init(shape=[self.num_domain, self.low_rank_dim], dtype=tf.float64),
                trainable=True)
            
            sms_init = tf.random_normal_initializer(0, 0.05)
            self.sms = tf.Variable( name = "sms",
                initial_value=sms_init(shape=[self.low_rank_dim+1, self.embedding_size, self.num_classes], dtype=tf.float64),
                trainable=True)
            
            sm_biases_init = tf.random_normal_initializer(0, 0.05)
            self.sm_biases = tf.Variable(name = "sm_biases",
                initial_value=sm_biases_init(shape=[self.low_rank_dim+1, self.num_classes], dtype=tf.float64),
                trainable=True)
            
        def csd(self, embeds, label_placeholder, domain_placeholder):    
            """CSD layer to be used as a replacement for the final classification layer
                Modified from: https://gist.github.com/vihari/bad9868049ef62db783e0fc11b22bb5c
            Args:
                embeds (tensor): final layer representations of dim 2
                label_placeholder (tensor): tf tensor with label index of dim 1
                domain_placeholder (tensor): tf tensor with domain index of dim 1 -- set to all zeros when testing
            Returns:
                tuple of final loss, logits
            """
            batch_size = tf.shape(embeds)[0]

            common_cwt = tf.identity(tf.concat([self.common_wt, tf.zeros([self.low_rank_dim])], axis=0), name='common_cwt')
            common_cwt = tf.cast(common_cwt, tf.float64)

            # Batch size x self.low_rank_dim + 1
            c_wts = tf.nn.embedding_lookup(self.emb_matrix, domain_placeholder)
            c_wts = tf.concat([ tf.cast(tf.ones([batch_size, 1]), tf.float64) * self.specialized_common_wt, c_wts], axis=1)
            c_wts = tf.reshape(c_wts, [batch_size, self.low_rank_dim+1])

            specific_sms = tf.einsum("ij,jkl->ikl", c_wts, self.sms)
            common_sm = tf.einsum("j,jkl->kl", common_cwt, self.sms)
            specific_bias = tf.einsum("ij,jl->il", c_wts, self.sm_biases)
            common_bias = tf.einsum("j,jl->l", common_cwt, self.sm_biases)

            diag_tensor = tf.eye(self.low_rank_dim+1, batch_shape=[self.num_classes], dtype=tf.float64)
            cps = tf.stack([tf.matmul(self.sms[:, :, _], self.sms[:, :, _], transpose_b=True) for _ in range(self.num_classes)])
            orthn_loss = tf.reduce_mean((cps - diag_tensor)**2)
            reg_loss = orthn_loss

            logits1 = tf.einsum("ik,ikl->il", embeds, specific_sms) + specific_bias
            logits2 = tf.matmul(embeds, common_sm) + common_bias

            loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=label_placeholder))
            loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=label_placeholder))

            loss = (1-self.common_specific_weights)*loss1 + self.common_specific_weights*loss2 + reg_loss

            # return common logits
            return loss, tf.math.softmax(logits2)

        def call(self, embeds, labels, domains):
            loss, logits = self.csd(embeds,labels,domains)
            self.add_loss(loss)
            return logits
    
    def build_csd_model(self):
        input_X = Input(shape = self.model_params["input_shape"], name = "input_X")
        if (self.model_params["flag_input_dict"]):
            if (self.model_params.get("flag_y_vector", True)):
                input_y = Input((2,), name="input_y")
            else:
                input_y = Input((), name="input_y")
            input_dataset = Input((), name="input_dataset", dtype = tf.int64)
            input_person = Input((), name="input_person", dtype = tf.int64)

        if (self.model_params["arch"] == "1dCNN"):
            feature_extractor = network.build_1dCNN(**self.model_params)
        elif (self.model_params["arch"] == "2dCNN"):
            feature_extractor = network.build_2dCNN(**self.model_params)
        elif (self.model_params["arch"] == "LSTM"):
            feature_extractor = network.build_LSTM(**self.model_params)
        elif (self.model_params["arch"] == "Transformer"):
            feature_extractor = network.build_Transformer(**self.model_params)
        
        embeddings = feature_extractor({"input_X":input_X, "input_y": input_y, "input_dataset":input_dataset, "input_person":input_person})

        csd_layer = self.csd_layer(embedding_size=self.model_params["embedding_size"],
            common_specific_weights = self.model_params["common_specific_weights"],
            num_domain = self.model_params["num_domain"],
            low_rank_dim = self.model_params["low_rank_dim"]
            )
        if (self.model_params["domain_target"] == "person"):
            preds = csd_layer(embeddings, input_y, input_person)
        elif (self.model_params["domain_target"] == "dataset"):
            preds = csd_layer(embeddings, input_y, input_dataset)
        model = Model(inputs=[input_X, input_y, input_dataset, input_person], outputs=[preds])
        return model

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(42)

        self.__assert__(X)

        model_optimizer = network.prep_model_optimizer(self.training_params)

        self.clf.compile(optimizer = model_optimizer, metrics="acc")

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
            df_results_record = self.fit_skip_training()

        return df_results_record

    def predict(self, X, y=None):
        self.__assert__(X)
        if (self.flag_X_dict):
            X_ = X["val_whole"] # only use the whole val set for eval
        else:
            X_ = X
        for data, label in X_:
            # zero out the domain
            if (self.model_params["domain_target"] == "person"):
                data["input_person"] = tf.zeros_like(data["input_person"])
            elif (self.model_params["domain_target"] == "dataset"):
                data["input_dataset"] = tf.zeros_like(data["input_dataset"])
            return np.argmax(self.clf.predict(data), axis = 1)

    def predict_proba(self, X, y=None):
        self.__assert__(X)
        if (self.flag_X_dict):
            X_ = X["val_whole"] # only use the whole val set for eval
        else:
            X_ = X
        for data, label in X_:
            # zero out the domain
            if (self.model_params["domain_target"] == "person"):
                data["input_person"] = tf.zeros_like(data["input_person"])
            elif (self.model_params["domain_target"] == "dataset"):
                data["input_dataset"] = tf.zeros_like(data["input_dataset"])
            return self.clf.predict(data)

class DepressionDetectionAlgorithm_DL_csd(DepressionDetectionAlgorithm_DL_erm):
    """ The CSD algorithm. Extends the ERM algorithm """

    def __init__(self, config_dict = None, config_name = "dl_csd"):
        super().__init__(config_dict, config_name)
        assert self.config["model_params"]["domain_target"] in ["person", "dataset"]
    
    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        self.config["model_params"].update(
            {"input_shape": self.input_shape,
            "flag_return_embedding":True, "flag_embedding_norm":False,
            "flag_input_dict":True}
        )
        if (self.config["model_params"]["domain_target"] == "person"):
            num_domain = len(self.data_generator_whole.person_dict)
        elif (self.config["model_params"]["domain_target"] == "dataset"):
            num_domain = len(self.data_generator_whole.dataset_dict)
        self.config["model_params"]["num_domain"] = num_domain

        return DepressionDetectionClassifier_DL_csd(config=self.config)