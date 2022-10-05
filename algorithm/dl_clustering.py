"""
Implementation of clustering-based Siamese network algorithm for depression detection

"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common_settings import *

from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm
from algorithm.dl_siamese import DepressionDetectionClassifier_DL_siamese
from data_loader.data_loader_ml import DataRepo
from utils import network, path_definitions, tf_metric_loss_64bit
from tensorflow.python.data.ops.dataset_ops import FlatMapDataset
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    Modified from: https://github.com/XifengGuo/DCEC

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, tdistribution_shape_param=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.tdistribution_shape_param = tdistribution_shape_param
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=tf.float64, shape=(None, input_dim))
        self.clusters = self.add_weight(name="clusters",shape = (self.n_clusters, input_dim),
            initializer='glorot_uniform', dtype=tf.float64)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.tdistribution_shape_param))
        q **= (self.tdistribution_shape_param + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# deep clustering model
class DCEC(object):
    '''
    Deep Clustering with Convolutional Autoencoders. Xifeng Guo, Xinwang Liu, En Zhu, Jianping Yin. ICONIP 2017
    Modified from https://github.com/XifengGuo/DCEC
    '''
    def __init__(self,
                 model_params, save_dir:str,
                 batch_size:int=128, verbose:int=0
                 ):

        super(DCEC, self).__init__()
        self.model_params = model_params
        self.verbose = verbose
        self.cae = network.build_autoencoder(**model_params)
        self.n_clusters = model_params["n_clusters"]
        self.clustering_loss_weight = model_params["loss_weight"]
        self.pretrained = False
        self.save_dir = save_dir
        self.dcec_weight_path = save_dir + "/dcec_model_pretrained.h5"
        self.cae_weight_path = os.path.split(save_dir)[0] + "/cae_model_pretrained.h5"
        self.y_pred = []
        hidden = self.cae.get_layer(name='embeddings').output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)
        self.batch_size = batch_size

        # Define DCEC model
        clustering_layer = ClusteringLayer(n_clusters=self.n_clusters,
            tdistribution_shape_param=model_params["tdistribution_shape_param"], name='clustering')(hidden)
        self.model = Model(inputs=self.cae.input,
                           outputs=[clustering_layer, self.cae.output])
        self.model.compile(loss=['kld', 'mse'], loss_weights=[self.clustering_loss_weight, 1],
                           optimizer = Adam(lr = 0.00005))

    def pretrain_cae(self, x, epochs=300):
        if (self.verbose > 0):
            print('...Pretraining...')
        if os.path.exists(self.cae_weight_path):
            self.cae.load_weights(self.cae_weight_path)
        else:
            self.cae.compile(optimizer = Adam(lr = 0.001), loss='mse')
            # begin training
            self.cae.fit(x, x, batch_size=self.batch_size, epochs=epochs,
                shuffle=True, verbose=0, callback = [tfa.callbacks.TQDMProgressBar()])
            self.cae.save(self.cae_weight_path)
            self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def fit(self, x, maxiter=2e4, tol=1e-3, shuffle = True,
            update_interval=200):

        if os.path.exists(self.dcec_weight_path):
            self.model.load_weights(self.dcec_weight_path)
            return

        idx_x = np.arange(len(x))
        if (shuffle):
            idx_x = np.random.permutation(idx_x)

        # Step 1: pretrain cae
        self.pretrain_cae(x)

        # Step 2: initialize cluster centers using k-means
        
        if (self.verbose > 0):
            print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                
                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float64) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)

                if (self.verbose > 1):
                    print('Iter', ite, '; loss=', loss, "; delta_label", delta_label)

                if ite > 0 and delta_label < tol:
                    if (self.verbose > 0):
                        print('delta_label ', delta_label, '< tol ', tol)
                        print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            if (index + 1) * self.batch_size > x.shape[0]:                    
                idx_range = idx_x[index * self.batch_size:]
                index = 0
                if (shuffle):
                    idx_x = np.random.permutation(idx_x)
            else:
                idx_range = idx_x[index * self.batch_size:(index + 1) * self.batch_size]
                index += 1

            loss = self.model.train_on_batch(x=x[idx_range],
                                             y=[p[idx_range],
                                                x[idx_range]])
            ite += 1
        # save the trained model
        if (self.verbose > 0):
            print('saving model to:', self.dcec_weight_path)
        self.model.save_weights(self.dcec_weight_path)

NAMARK = tf.cast(float('inf'), tf.float64)
class DepressionDetectionClassifier_DL_clustering(DepressionDetectionClassifier_DL_siamese):
    """ Clustering + Ensemble Siamese network. Extended from Siamese Claasifer """

    def __init__(self, config):
        super().__init__(config=config)

        self.clustering_model_params = self.model_params["clustering"]
        self.clustering_model_params["input_shape"] = self.model_params["input_shape"]
        self.n_clusters = self.clustering_model_params["n_clusters"]

        self.clf_dict = {}
        self.embeddings_train_repo_dict = {}
        self.y_train_repo_dict = {}

        if (self.model_params["arch"] == "1dCNN"):
            self.clf_func = network.build_1dCNN
        elif (self.model_params["arch"] == "2dCNN"):
            self.clf_func = network.build_2dCNN
        elif (self.model_params["arch"] == "LSTM"):
            self.clf_func = network.build_LSTM
        elif (self.model_params["arch"] == "Transformer"):
            self.clf_func = network.build_Transformer

        for cluster_idx in range(self.n_clusters):
            self.clf_dict[cluster_idx] = self.clf_func(**self.model_params)
        self.clustering_model_params.update({
            "flag_return_embedding":False, "flag_embedding_norm":False,
            "flag_input_dict":False})

        # save folder
        self.save_folder = os.path.join(path_definitions.TMP_PATH, "clustering",
            "input_shape-" + "_".join([str(i) for i in self.model_params["input_shape"]]),
            self.config["data_loader"]["training_dataset_key"],
            "cluster_" + str(self.n_clusters),
        )
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        
        self.dcec = DCEC(model_params=self.clustering_model_params, save_dir=self.save_folder,
            batch_size=self.config["data_loader"]["batch_size"], verbose=self.training_params["clustering_verbose"])
        self.dcec_weight_path = os.path.join(self.save_folder, "dcec_model_pretrained.h5")

        # TODO: can be improved
        if ("focus_cluster_idx" in self.training_params and 
            len(self.training_params["focus_cluster_idx"]) > 0):
            self.flag_partial_cluster = True
        else:
            self.flag_partial_cluster = False

    class tiny_data_generator():
        """Generator wrapper for clustering"""
        def __init__(self, model_params, X, y, person, dataset, is_training, batch_size = 64):
            self.model_params=model_params
            self.X = X
            self.y = y
            self.is_training = is_training
            self.person = person
            self.dataset = dataset
            self.batch_size = batch_size

            if (self.y is not None):
                self.tf_output_signature = ({
                        "input_X": tf.TensorSpec(shape=[None] + self.model_params["input_shape"], dtype = tf.float64),
                        "input_y": tf.TensorSpec(shape=(None, 2) if self.model_params["flag_y_vector"] else (None), dtype = tf.float64),
                        "input_dataset": tf.TensorSpec(shape=(None), dtype = tf.int64),
                        "input_person": tf.TensorSpec(shape=(None), dtype = tf.int64),
                    }, tf.TensorSpec(shape=(None, 2) if self.model_params["flag_y_vector"] else (None), dtype = tf.float64))
                self.sample_size = len(self.y)
                self.index_list = np.arange(self.sample_size)
                self.step_per_epoch = int(np.ceil(self.sample_size / self.batch_size))
            else:
                self.tf_output_signature = ({
                        "input_X": tf.TensorSpec(shape=(None), dtype = tf.float64),
                        "input_y": tf.TensorSpec(shape=(None), dtype = tf.float64),
                        "input_dataset": tf.TensorSpec(shape=(None), dtype = tf.float64),
                        "input_person": tf.TensorSpec(shape=(None), dtype = tf.float64),
                    }, tf.TensorSpec(shape=(None), dtype = tf.float64))

        def __call__(self):
            while True:
                if (self.y is not None):
                    if (self.is_training):
                        batch_count = 0
                        self._shuffle()
                        while batch_count < self.step_per_epoch:
                            idx_start = batch_count * self.batch_size
                            idx_end = idx_start + self.batch_size               
                            idx = self.index_list[idx_start: idx_end]
                            yield {"input_X":tf.gather(self.X,idx,axis=0),
                                "input_y":tf.gather(self.y,idx,axis=0),
                                "input_dataset": tf.gather(self.dataset,idx,axis=0),
                                "input_person": tf.gather(self.person,idx,axis=0)}, tf.gather(self.y,idx,axis=0)
                    else:
                        yield {"input_X":self.X, "input_y":self.y, "input_dataset": self.dataset, "input_person": self.person}, self.y
                        break
                else:
                    yield {"input_X":[NAMARK], "input_y":[NAMARK], "input_dataset": [NAMARK], "input_person": [NAMARK]}, [NAMARK]
        
        def _shuffle(self):
            np.random.shuffle(self.index_list)
        
    def inspect_cluster(self, data: Dict[str, FlatMapDataset],
        n_clusters = 10, confidence_min_th = None, is_training = False):
        cluster_prob = self.dcec.model.predict(data["input_X"])[0]
        cluster_pred = np.argmax(cluster_prob, axis=1)
        if (confidence_min_th is not None):
            idx_above_th = set(np.where(np.max(cluster_prob, axis=1) > confidence_min_th)[0])
        
        cluster_idx = 0
        cluster_data_idx_dict = {}
        for cluster_idx in range(n_clusters):
            data_idx = np.where(cluster_pred == cluster_idx)[0]
            if (confidence_min_th is not None):
                data_idx = [i for i in data_idx if i in idx_above_th]
            cluster_data_idx_dict[cluster_idx] = deepcopy(data_idx)
            if (self.training_params["verbose"] > 0):
                print(cluster_idx, len(data_idx), end = ", ")
        if (self.training_params["verbose"] > 0):
            print("", end = "\n") # line break

        user_repo_cluster_dict = {}
        for cluster_idx, data_idx in cluster_data_idx_dict.items():
            user_repo_cluster_dict[cluster_idx] = set(data["input_person"].numpy()[data_idx])

        user_in_cluster_dict = {}
        for user in set(data["input_person"].numpy()):
            user_in_cluster_dict[user] = 0
            for cluster_idx, user_repo in user_repo_cluster_dict.items():
                if (user in user_repo):
                    user_in_cluster_dict[user] += 1
                    
        data_repo_cluster_dict = {}
        for cluster_idx in range(n_clusters):
            data_idx = cluster_data_idx_dict[cluster_idx]
            if (len(data_idx) > 0):
                X_tmp = tf.gather(data["input_X"], data_idx, axis = 0)
                y_tmp = tf.gather(data["input_y"], data_idx, axis = 0)
                person_tmp = tf.gather(data["input_person"], data_idx, axis = 0)
                dataset_tmp = tf.gather(data["input_dataset"], data_idx, axis = 0)
            else:
                X_tmp, y_tmp, person_tmp, dataset_tmp = None, None, None, None
            data_gen = self.tiny_data_generator(model_params=self.model_params,
                X = X_tmp, y = y_tmp,
                person = person_tmp, dataset = dataset_tmp,
                is_training=is_training, batch_size=self.config["data_loader"]["cluster_batch_size"])
            data_repo_cluster_dict[cluster_idx] = tf.data.Dataset.from_generator(data_gen,
                            output_signature=data_gen.tf_output_signature)

        return cluster_pred, cluster_data_idx_dict, user_repo_cluster_dict, user_in_cluster_dict, data_repo_cluster_dict

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(42)

        self.__assert__(X)
        assert self.flag_X_dict

        # Training the deep clustering model
        for data, label in X["val_whole"] if self.flag_X_dict else X: break
        self.dcec.fit(data["input_X"].numpy(), tol=self.clustering_model_params["error_tolerance"], maxiter=20000,
                    update_interval=200)
        
        data_cluster_dict = {}
        for key in [k for k in X.keys() if k != "train"]:
            for data, label in X[key] if self.flag_X_dict else X: break
            cluster_pred, cluster_data_idx_dict, user_repo_cluster_dict, user_in_cluster_dict, data_cluster_dict[key] =\
                self.inspect_cluster(data, n_clusters=self.n_clusters, is_training= True if key == "train" else False)
        for data, label in X["valtrain"] if self.flag_X_dict else X: break
        cluster_pred, cluster_data_idx_dict, user_repo_cluster_dict, user_in_cluster_dict, data_cluster_dict["train"] =\
            self.inspect_cluster(data, n_clusters=self.n_clusters, is_training= True)
            
        X_cluster_dict = {}
        for cluster_idx in range(self.n_clusters):
            for d_train, l_train in data_cluster_dict["train"][cluster_idx]: break
            if (len(l_train) == 1):
                continue
            X_cluster_dict[cluster_idx] = {
                "train": data_cluster_dict["train"][cluster_idx],
                "val": data_cluster_dict["val"][cluster_idx],
                "val_whole": data_cluster_dict["valtrain"][cluster_idx], # t-sne could have different cluster assignment after adding val. So only use train data
            }
            if "test" in data_cluster_dict:
                X_cluster_dict[cluster_idx]["test"] = data_cluster_dict["test"][cluster_idx]
        df_results_record_dict = {}

        if (self.flag_partial_cluster):
            X_cluster_dict_new = {k:X_cluster_dict[k] for k in
                list(set(self.training_params["focus_cluster_idx"]).intersection(set(X_cluster_dict.keys())))}
        else:
            X_cluster_dict_new = X_cluster_dict
        for cluster_idx, X_cluster in X_cluster_dict_new.items():
            if (self.training_params["verbose"] > 0):
                print("current cluster idx:", cluster_idx)
            self.current_cluster_idx = cluster_idx
            self.clf = self.clf_func(**self.model_params)
            self.clf.set_weights(deepcopy(self.clf_dict[cluster_idx].get_weights()))
            model_optimizer = network.prep_model_optimizer(self.training_params)

            self.clf.compile(loss = tf_metric_loss_64bit.TripletSemiHardLoss_64bit(margin=self.model_params["triplet_loss_margin"]),
                optimizer = model_optimizer)

            for d_val, l_val in X_cluster["val"]: break
            for d_val_whole, l_val_whole in X_cluster["val_whole"]: break
            flag_null_val = True if d_val["input_y"][0] == NAMARK else False
            flag_null_val_whole = True if d_val_whole["input_y"][0] == NAMARK else False

            if "test" in X_cluster:
                for d_test, l_test in X_cluster["test"]: break
                flag_null_test = True if d_test["input_y"][0] == NAMARK else False
            else:
                flag_null_test = True
            callback_dskeys = list(X_cluster.keys())
            if (flag_null_test):
                callback_dskeys = [k for k in callback_dskeys if k != "test"]

            callbacks = self.prep_callbacks({k: X_cluster[k] for k in callback_dskeys})

            if (self.training_params.get("skip_training", False) == False):
                history = self.clf.fit(x = X_cluster["train"],
                        steps_per_epoch = int(np.ceil(len(l_val_whole) / self.config["data_loader"]["cluster_batch_size"])),
                        epochs = self.training_params["epochs"],
                        validation_data = (d_val, l_val) if not flag_null_val else None,
                        verbose = 1 if self.training_params["verbose"] > 1 else 0,
                        callbacks = callbacks
                        )
                self.log_history = history.history

                best_epoch, df_results_record = self.find_best_epoch()
                self.clf.set_weights(self.model_saver.model_repo_dict[best_epoch])
            else:
                df_results_record = self.fit_skip_training()

            df_results_record_dict[cluster_idx] = df_results_record
            
            self.clf_dict[cluster_idx].set_weights(deepcopy(self.clf.get_weights()))

            self.embeddings_train_repo_dict[cluster_idx] = deepcopy(self.clf_dict[cluster_idx].predict(X_cluster["val_whole"]))
            self.y_train_repo_dict[cluster_idx] = np.array(l_val_whole)

        return df_results_record_dict

    def cluster_predict(self, data, label, flag_is_train):
        total_len = len(label)
        # cluster_prob = self.dcec.model.predict(data["input_X"])[0]
        cluster_pred, cluster_data_idx_dict, user_repo_cluster_dict, user_in_cluster_dict, data_cluster_dict =\
            self.inspect_cluster(data, n_clusters=self.n_clusters)
        y_pred_merged = np.zeros(total_len)
        y_pred_prob_merged = np.zeros((total_len, 2))
        record_tmp = {}
        for cluster_idx, X_cluster in data_cluster_dict.items():
            for data_cluster, label_cluster in X_cluster: break
            if data_cluster["input_y"][0] == NAMARK: continue
            if (cluster_idx not in self.clf_dict or cluster_idx not in self.embeddings_train_repo_dict or cluster_idx not in self.y_train_repo_dict):
                y_pred = np.array([0 for _ in range(len(data_cluster["input_y"]))])
                y_pred_prob = np.array([[1,0] for _ in range(len(data_cluster["input_y"]))])
            else:
                y_pred, y_pred_prob = self.single_predict(self.clf_dict[cluster_idx], data_cluster,
                    self.embeddings_train_repo_dict[cluster_idx], self.y_train_repo_dict[cluster_idx], flag_is_train)
            record_tmp[cluster_idx] = (y_pred, y_pred_prob)
            for data_idx, y_pred_single, y_pred_prob_single in zip(cluster_data_idx_dict[cluster_idx], y_pred, y_pred_prob):
                y_pred_merged[data_idx] = y_pred_single
                y_pred_prob_merged[data_idx] = y_pred_prob_single
        return y_pred_merged, y_pred_prob_merged

    def predict(self, X, y=None):
        self.__assert__(X)
        if (self.flag_X_dict):
            X_ = X["val_whole"] # only use the whole val set for eval
            flag_is_train = True
        else:
            X_ = X
            flag_is_train = False
        for data, label in X_:
            if (self.flag_partial_cluster):
                # give up prediction TODO: can be improved
                y_pred = np.zeros(len(label))
            else:
                y_pred, y_pred_prob = self.cluster_predict(data, label, flag_is_train)
            return y_pred 

    def predict_proba(self, X, y=None):
        self.__assert__(X)
        if (self.flag_X_dict):
            X_ = X["val_whole"] # only use the whole val set for eval
            flag_is_train = True
        else:
            X_ = X
            flag_is_train = False
        for data, label in X_:
            if (self.flag_partial_cluster):
                # give up prediction TODO: can be improved
                y_pred_prob = np.array([[1,0] for i in range(len(label))])
            else:
                y_pred, y_pred_prob = self.cluster_predict(data, label, flag_is_train)
            return y_pred_prob

class DepressionDetectionAlgorithm_DL_clustering(DepressionDetectionAlgorithm_DL_erm):
    """ A deep clustering based algorithm. Extends the ERM algorithm """

    def __init__(self, config_dict = None, config_name = "dl_clustering"):
        super().__init__(config_dict, config_name)
    
    def prep_model(self, data_train: DataRepo, criteria: str = "balanced_acc") -> sklearn.base.ClassifierMixin:
        self.config["model_params"].update(
            {"input_shape": self.input_shape,
            "flag_return_embedding":True, "flag_embedding_norm":True,
            "flag_input_dict":True}
        )
        return DepressionDetectionClassifier_DL_clustering(config = self.config)