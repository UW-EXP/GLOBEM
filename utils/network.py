import sys
from typing import List, Dict
sys.path.append("../")
sys.path.append("../../")
from common_settings import *
from tensorflow.python.data.ops.dataset_ops import FlatMapDataset

#### Model Architecture Definition

def build_1dCNN(input_shape: List[int],
    conv_shapes:List[int] = [8,4], embedding_size:int = 8,
    flag_return_embedding:bool = False, flag_embedding_norm = False,
    flag_input_dict:bool = True, flag_y_vector:bool = True,
    **kwargs_holder) -> Model:
    """Define a 1dCNN model"""

    input_X = Input(shape = input_shape, name = "input_X")
    if (flag_input_dict):
        if (flag_y_vector):
            input_y = Input((2,), name="input_y")
        else:
            input_y = Input((), name="input_y")
        input_dataset = Input((), name="input_dataset", dtype = tf.int64)
        input_person = Input((), name="input_person", dtype = tf.int64)
    
    x = input_X
    
    for i, conv_shape in enumerate(conv_shapes):
        x = Conv1D(conv_shape, (3), activation='relu', strides = 1,
                    kernel_initializer='he_uniform', padding = "same",
                    kernel_regularizer=l2(2e-4))(x)
        x = BatchNormalization()(x)
        if (i < 2):
            x = MaxPooling1D()(x)
        x = Dropout(0.25)(x)
    x = Flatten()(x)
    embeddings = Dense(embedding_size, activation = "relu", name = "embeddings")(x)
    if (not flag_return_embedding):
        out = Dense(2, activation = "softmax")(embeddings)
    else:
        if (flag_embedding_norm):
            out = Lambda(lambda y: K.l2_normalize(y,axis=-1), name = "normalization")(embeddings)
        else:
            out = embeddings
    
    if (flag_input_dict):
        return Model(inputs = [input_X, input_y, input_dataset, input_person], outputs = [out])
    else:
        return Model(inputs = [input_X], outputs = [out])

def build_2dCNN(input_shape: List[int],
    conv_shapes:List[int] = [8,4], embedding_size:int = 8,
    flag_return_embedding:bool = False, flag_embedding_norm = False,
    flag_input_dict:bool = True, flag_y_vector:bool = True,
    **kwargs_holder) -> Model:
    """Define a 2dCNN model"""

    input_X = Input(shape = input_shape, name = "input_X")
    if (flag_input_dict):
        if (flag_y_vector):
            input_y = Input((2,), name="input_y")
        else:
            input_y = Input((), name="input_y")
        input_dataset = Input((), name="input_dataset", dtype = tf.int64)
        input_person = Input((), name="input_person", dtype = tf.int64)
    
    x = input_X
    
    for i, conv_shape in enumerate(conv_shapes):
        x = Conv2D(conv_shape, (3), activation='selu', strides = 1,
                    kernel_initializer='he_uniform', padding = "same",
                    kernel_regularizer=l2(2e-4))(x)
        x = BatchNormalization()(x)
        if (i < 2):
            x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)
    x = Flatten()(x)
    embeddings = Dense(embedding_size, activation = "selu", name = "embeddings")(x)
    if (not flag_return_embedding):
        out = Dense(2, activation = "softmax")(embeddings)
    else:
        if (flag_embedding_norm):
            out = Lambda(lambda y: K.l2_normalize(y,axis=-1))(embeddings)
        else:
            out = embeddings
    
    if (flag_input_dict):
        return Model(inputs = [input_X, input_y, input_dataset, input_person], outputs = [out])
    else:
        return Model(inputs = [input_X], outputs = [out])

def build_LSTM(input_shape: List[int],
    lstm_shapes:List[int] = [20], embedding_size:int = 8,
    flag_bidirection:bool = False, flag_allsequences:bool = False,
    flag_return_embedding:bool = False, flag_embedding_norm = False,
    flag_input_dict:bool = True, flag_y_vector:bool = True,
    **kwargs_holder) -> Model:
    """Define a LSTM model"""

    input_X = Input(shape = input_shape, name = "input_X")
    if (flag_input_dict):
        if (flag_y_vector):
            input_y = Input((2,), name="input_y")
        else:
            input_y = Input((), name="input_y")
        input_dataset = Input((), name="input_dataset", dtype = tf.int64)
        input_person = Input((), name="input_person", dtype = tf.int64)
    
    x = input_X
    
    for lstm_shape in lstm_shapes[:-1]:
        if flag_bidirection:
            x = Bidirectional(LSTM(lstm_shape, return_sequences = True,
                    kernel_initializer='he_uniform',
                    kernel_regularizer=l2(2e-4)))(x)
        else:
            x = LSTM(lstm_shape, return_sequences = True,
                    kernel_initializer='he_uniform',
                    kernel_regularizer=l2(2e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

    if flag_bidirection:
        x = Bidirectional(LSTM(lstm_shapes[-1], return_sequences = flag_allsequences,
                kernel_initializer='he_uniform',
                kernel_regularizer=l2(2e-4)))(x)
    else:
        x = LSTM(lstm_shapes[-1], return_sequences = flag_allsequences,
                kernel_initializer='he_uniform',
                kernel_regularizer=l2(2e-4))(x)
    if(flag_allsequences):
        x = Lambda(lambda y: K.mean(y, axis = -2))(x)
    
    x = Flatten()(x)
    
    embeddings = Dense(embedding_size, activation = "selu", name = "embeddings")(x)
    if (not flag_return_embedding):
        out = Dense(2, activation = "softmax")(embeddings)
    else:
        if (flag_embedding_norm):
            out = Lambda(lambda y: K.l2_normalize(y,axis=-1))(embeddings)
        else:
            out = embeddings
    
    if (flag_input_dict):
        return Model(inputs = [input_X, input_y, input_dataset, input_person], outputs = [out])
    else:
        return Model(inputs = [input_X], outputs = [out])

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0)  -> tf.keras.layers:
    """Define a transformer block"""
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_Transformer(
    input_shape: List[int],
    head_size:int = 4, num_heads:int = 4, ff_dim:int = 16,
    num_transformer_blocks:int = 2, transfromer_dropout:float = 0,
    mlp_units:List[int] = [32], mlp_dropout:float=0,
    embedding_size:int = 8,
    flag_return_embedding:bool = False, flag_embedding_norm:bool = False,
    flag_input_dict:bool = True, flag_y_vector:bool = True,
    **kwargs_holder) -> Model:
    """Define a transformer model"""
    
    input_X = Input(shape = input_shape, name = "input_X")
    if flag_input_dict:
        if (flag_y_vector):
            input_y = Input((2,), name="input_y")
        else:
            input_y = Input((), name="input_y")
        input_dataset = Input((), name="input_dataset", dtype = tf.int64)
        input_person = Input((), name="input_person", dtype = tf.int64)
    
    x = input_X
    # x = BatchNormalization(momentum = 0)(x)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, transfromer_dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)

    embeddings = Dense(embedding_size, activation = "selu", name = "embeddings")(x)
    if (not flag_return_embedding):
        out = Dense(2, activation = "softmax")(embeddings)
    else:
        if (flag_embedding_norm):
            out = Lambda(lambda y: K.l2_normalize(y,axis=-1))(embeddings)
        else:
            out = embeddings
    if flag_input_dict:
        return keras.Model([input_X, input_y, input_dataset, input_person], [out])
    else:
        return keras.Model([input_X], [out])


def build_autoencoder(input_shape:List[int],
    conv_shapes:List[int] = [8,4], embedding_size:int = 8,
    flag_input_dict:bool = True, flag_y_vector:bool = True,
    **kwargs_holder) -> Model:
    """Define a autoencoder-decoder"""

    input_X = Input(shape = input_shape, name = "input_X")

    if flag_input_dict:
        if (flag_y_vector):
            input_y = Input((2,), name="input_y")
        else:
            input_y = Input((), name="input_y")
        input_dataset = Input((), name="input_dataset", dtype = tf.int64)
        input_person = Input((), name="input_person", dtype = tf.int64)

    if (len(input_shape) == 2):
        d_width = input_shape[0]
        d_height = None
        conv_func = Conv1D
        conv_transpoe_func = Conv1DTranspose
        maxpool_func = MaxPooling1D
        uppool_func = UpSampling1D
    elif (len(input_shape) == 3):
        d_width = input_shape[0]
        d_height = input_shape[1]
        conv_func = Conv2D
        conv_transpoe_func = Conv2DTranspose
        maxpool_func = MaxPooling2D
        uppool_func = UpSampling2D

    x = input_X

    for i, conv_shape in enumerate(conv_shapes):
        x = conv_func(conv_shape, (5), activation='selu', strides = 1,
                    kernel_initializer='he_uniform', padding = "same",
                    kernel_regularizer=l2(2e-4))(x)
        if (i < 3):
            x = maxpool_func()(x)
            d_width = int(round(d_width / 2))
            if (d_height):
                d_height = int(round(d_height / 2))
    d_depth = conv_shapes[-1]
    x = Flatten()(x)
    embeddings = Dense(embedding_size, name = "embeddings")(x)
    if (len(input_shape) == 2):
        x_ = Dense(d_width * d_depth)(embeddings)
        x_ = Reshape((d_width, d_depth))(x_)
    elif (len(input_shape) == 3):
        x_ = Dense(d_width * d_height * d_depth)(embeddings)
        x_ = Reshape((d_width, d_height, d_depth))(x_)
    for i, conv_shape in zip(list(range(len(conv_shapes)))[::-1], conv_shapes[::-1]):
        if (i < 3):
            x_ = uppool_func()(x_)
        x_ = conv_transpoe_func(conv_shape, (5), activation='selu', strides = 1,
                        kernel_initializer='he_uniform', padding = "same",
                        kernel_regularizer=l2(2e-4))(x_)
    
    x_ = conv_transpoe_func(input_shape[-1], (5), activation='selu', strides = 1,
                    kernel_initializer='he_uniform', padding = "same",
                    kernel_regularizer=l2(2e-4))(x_)
    if (len(input_shape) == 3):
        # crop if necessary
        new_height = x_.shape[2]
        delta_height = new_height - input_shape[1]
        crop_up = int(delta_height / 2)
        crop_bottom = delta_height - crop_up
        x_ = Cropping2D(((0,0),(crop_up, crop_bottom)))(x_)

    out = x_
    
    if (flag_input_dict):
        return Model(inputs = [input_X, input_y, input_dataset, input_person], outputs = [out])
    else:
        return Model(inputs = [input_X], outputs = [out])

def prep_model_optimizer(training_params: Dict[str, float]):
    """Define a tensorflow optimizer"""
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=training_params["learning_rate"],
            first_decay_steps=training_params["cos_annealing_step"],
            m_mul=training_params["cos_annealing_decay"],
            alpha=0.01, t_mul=2)
    if training_params["optimizer"] == "SGD":
        model_optimizer = tf.optimizers.SGD(lr_decayed_fn)
    elif training_params["optimizer"] == "Adam":
        model_optimizer = tf.optimizers.Adam(lr_decayed_fn)
    return model_optimizer

#### Callback functions

class EvaluationBasicCallback(Callback):
    """A callback function to evaluate model performance on a pre-defined datasets"""
    def __init__(self, model_obj: Model,
            dataset_train:FlatMapDataset, dataset_test:FlatMapDataset=None,
            interval:int=1, verbose:int=1, flag_skip_y_defition:bool=False):
        super(Callback, self).__init__()
        self.interval = interval
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        if (self.dataset_test is not None):
            self.flag_with_test = True
        else:
            self.flag_with_test = False
        if (not flag_skip_y_defition):
            self.y_train = np.array([i[1] for i in self.dataset_train][0])
            if (len(self.y_train.shape) > 1): # if y is a vector, convert to sparse
                self.y_train = np.argmax(self.y_train, axis = 1)
            
            if (self.flag_with_test):
                self.y_test = np.array([i[1] for i in self.dataset_test][0])
                if (len(self.y_test.shape) > 1): # if y is a vector, convert to sparse
                    self.y_test = np.argmax(self.y_test, axis = 1)
        self.results_record = []
        self.model_obj = model_obj
        self.verbose = verbose

    def process_results(self, epoch, logs, results_train, results_test):
        results_train = {k+"_train":v for k,v in results_train.items()}
        results = results_train
        if (results_test is not None):
            results_test = {k+"_test":v for k,v in results_test.items()}
            results.update(results_test)
        if (logs is not None):
            logs_dict = {"logs_" + k: v for k,v in logs.items()}
        else:
            logs_dict = {}
        logs_dict["epoch"] = epoch
        results.update(logs_dict)

        if (self.verbose > 0):
            print(f"Epoch {epoch} -- Train: acc - {results['acc_train']:.3f} balacc - {results['balanced_acc_train']:.3f} auc - {results['roc_auc_train']:.3f}",
                f"cfmtx - {results['cfmtx_train']}")
            if (results_test is not None):
                print(f"Epoch {epoch} -- Val: acc - {results['acc_test']:.3f} balacc - {results['balanced_acc_test']:.3f} auc - {results['roc_auc_test']:.3f}",
                    f"cfmtx - {results['cfmtx_test']}",
                    )
        self.results_record.append(results)

    def on_epoch_end(self, epoch, logs=None):
        """ basic process function, can be overwritten """
        if (epoch % self.interval == 0):
            results_train = utils_ml.results_report_sklearn(clf = self.model_obj,
                X=self.dataset_train, y=self.y_train, return_confusion_mtx=True)
            if (self.flag_with_test):
                results_test = utils_ml.results_report_sklearn(clf = self.model_obj,
                    X=self.dataset_test, y=self.y_test, return_confusion_mtx=True)
            else:
                results_test = None
            self.process_results(epoch, logs, results_train, results_test)

class ModelMemoryCheckpoint(Callback):
    """A callback function to manually save model weights at every interval"""
    def __init__(self, interval = 1):
        self.model_repo_dict = {}
        self.interval = interval
    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.interval == 0):
            self.model_repo_dict[epoch] = deepcopy(self.model.get_weights())

