import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tqdm
import mitdeeplearning as mdl
from statistics import mean


def read_data(file_path):
    data = mne.io.read_raw_edf(file_path)
    raw_data = data.get_data()
    EEG_Cz = raw_data[0] * 1000000
    EEG_Oz = raw_data[1] * 1000000
    return EEG_Cz, EEG_Oz


def get_batch(original_data, seq_length, batch_size):
    # the length of the vectorized songs string
    n = original_data.shape[0] - 1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n - seq_length, batch_size)

    # construct a list of input sequences for the training batch
    input_batch = [original_data[i:i + seq_length] for i in idx]
    # construct a list of output sequences for the training batch
    output_batch = [original_data[i + 1:i + seq_length + 1] for i in idx]

    # x_batch, y_batch provide the true inputs and targets for network training
    X_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return X_batch, y_batch


def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )


def build_model(n_bins, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(n_bins, embedding_dim, batch_input_shape=[batch_size, None]),
    LSTM(rnn_units),
    tf.keras.layers.Dense(units = n_bins)
    ])
    return model


def compute_loss(y_truth, y_hat):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_truth, y_hat, from_logits=True) # TODO
    return loss


def fit(x, y, optimizer, model):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def evaluate(X_test, y_test, model):
    y_hat = model(X_test)
    loss = compute_loss(y_test, y_hat)
    #predictions = tf.squeeze(y_hat, 0)
    #predict_tmp = np.argmin(predictions[-1,:])
    #predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
    #y_truth = y_test[0, -1]
    return loss.numpy().mean()


def main():
    print("Begin main")
    # subject 0, night 0
    SEQ_LEN = 100
    BATCH_SIZE = 64
    file_path = "dataset\sleep-cassette\SC4001E0-PSG.edf"
    EEG_Cz, EEG_Oz = read_data(file_path)
    test_size = len(EEG_Cz)*0.2
    EEG_Cz_train = EEG_Cz[0: -int(test_size)]
    EEG_Cz_test = EEG_Cz[-int(test_size): -1]

    EEG_Cz_train = np.reshape(EEG_Cz_train, (-1, 1))
    EEG_Cz_test = np.reshape(EEG_Cz_test, (-1, 1))

    max_voltage = np.max(EEG_Cz)
    min_voltage = np.min(EEG_Cz)
    scaler = MinMaxScaler()
    EEG_Cz_train = scaler.fit_transform(EEG_Cz_train)
    EEG_Cz_test = scaler.transform(EEG_Cz_test)

    # pre-process the raw EEG readings: quantization
    n_bins = 3620 # predict at precision of .1 micro volt
    quantizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    EEG_Cz_train = quantizer.fit_transform(EEG_Cz_train)
    EEG_Cz_test = quantizer.transform(EEG_Cz_test)

    rnn_units = 1024 # hyperparams to be tuned
    embedding_dim = 256 # hyperparams to be tuned
    learning_rate = 5e-3 # hyperparams to be tuned
    epochs = 2000
    model = build_model(n_bins=n_bins, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)
    optimizer = Adam(learning_rate)
    model.summary()

    """
    checkpoint_dir = './training_checkpoints'

    best_loss = 10
    best_weight_file = ""

    # fit the model
    for i in range(epochs):
        X_batch, y_batch = get_batch(original_data=EEG_Cz_train, seq_length=SEQ_LEN, batch_size=BATCH_SIZE)
        loss = fit(X_batch, y_batch, optimizer, model)
        loss_mean = loss.numpy().mean()
        if loss_mean < best_loss:
            best_loss = loss_mean
            print("Best loss updated at epoch #", i, ":", best_loss)
            model.save_weights(os.path.join(checkpoint_dir, "best_loss" + "_epoch_" + str(i)))
            best_weight_file = os.path.join(checkpoint_dir, "best_loss" + "_epoch_" + str(i))
        else:
            print("Epoch #", i, "; loss:", loss_mean)
        if i % 1000 == 0:
            model.save_weights(os.path.join(checkpoint_dir, str(i)))
    """
    model = build_model(n_bins, embedding_dim, rnn_units, batch_size=1)
    best_weight_file = "./training_checkpoints/best_loss_epoch_1698"
    # evaluate model
    model.load_weights(best_weight_file)
    loss_list = []
    for i in range(1000):
        X_batch, y_batch = get_batch(original_data=EEG_Cz_test, seq_length=SEQ_LEN, batch_size=1)
        loss_list.append(evaluate(X_batch, y_batch, model))
    print("Loss on testing set: ", mean(loss_list))


if __name__ == '__main__':
    main()
