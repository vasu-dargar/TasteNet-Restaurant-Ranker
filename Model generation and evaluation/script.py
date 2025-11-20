
import warnings
import tensorflow as tf
from tensorflow import keras
import argparse
import os
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, constraints, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class AttentionLayer(layers.Layer):

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = keras.initializers.get('glorot_uniform')

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.W_constraint = keras.constraints.get(W_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(AttentionLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True


    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None


    def call(self, x, mask=None):
        # TF backend doesn't support it
        # eij = K.dot(x, self.W) 
        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), 
                              K.reshape(self.W, (features_dim, 1))),
                        (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


    def get_config(self):
        config = {'step_dim': self.step_dim}
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Directories provided by SageMaker
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    # ---- custom hyperparameters from SageMaker ----
    parser.add_argument("--max-seq-length", type=int, default=100)
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--gru-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)

    args, _ = parser.parse_known_args()

    print("[INFO] Reading data\n")
    X_path = os.path.join(args.train, "X_train_V_1.csv")
    y_path = os.path.join(args.train, "y_train_V_1.csv")

    X_train = pd.read_csv(X_path)
    y_train = pd.read_csv(y_path)

    X_train=X_train.to_numpy()
    y_train=y_train.to_numpy()

    MAX_SEQUENCE_LENGTH = args.max_seq_length
    VOCAB_SIZE = args.vocab_size
    EMBEDDING_DIM = args.embedding_dim
    GRU_DIM = args.gru_dim
    batch_size = args.batch_size
    epochs = args.epochs
    
    print("[INFO] Reading completed")

    sequence_input = layers.Input(shape=(MAX_SEQUENCE_LENGTH,))

    print("Sequence input layer created")

    embedded_sequences = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, trainable=True)(sequence_input)

    print("Embedding of sequence input layer done")

    #bigru = Bidirectional(layers.CuDNNGRU(GRU_DIM, return_sequences=True))(embedded_sequences)
    bigru = layers.Bidirectional(layers.GRU(GRU_DIM, return_sequences=True, recurrent_dropout=0.2))(embedded_sequences)

    print("Bi-GRU added")

    att = AttentionLayer(MAX_SEQUENCE_LENGTH)(bigru)

    print("Attention Layer added")

    dense1 = layers.Dense(GRU_DIM*2, activation='relu')(att)

    print("1st dense layer created")

    dropout1 = layers.Dropout(rate=0.3)(dense1)

    print("Intitiated drop")

    dense2= layers.Dense(GRU_DIM, activation='relu')(dropout1)

    print("2nd dense layer created")

    dropout2 = layers.Dropout(rate=0.3)(dense2)

    print("Intitiated drop")

    outp = layers.Dense(5, activation='linear')(dense2)

    print("Output layer created")

    # initialize the model

    model = keras.Model(inputs=sequence_input, outputs=outp)

    print("Model created")

    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    print("Model compiled")

    model.summary()

    # ----------------------------
    # Training
    # ----------------------------
    
    output_data_dir=args.output_data_dir
    os.makedirs(output_data_dir, exist_ok=True)

    chk_pt=os.path.join(output_data_dir,"Epoch_{epoch:04d}_.weights.h5")

    mc = ModelCheckpoint(chk_pt, verbose=1, save_weights_only=True, save_freq='epoch')

    print("Created an utility for best model checkpoint")

    history = model.fit(X_train, y_train, batch_size=batch_size, shuffle=True, validation_split=0.1, epochs=epochs, verbose=1, callbacks=[mc])

    model_dir=args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "1")
    model.export(model_path)