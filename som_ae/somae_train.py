import pickle as pickle
import numpy as np
import tensorflow as tf

from somae_model import SOMAE
import pandas as pd

# data_set = "/home/aritacco/SOM_AE/SOM-VAE/data/training_dataset.obj"
dataset_train = pd.read_csv('../data/HAPT_Dataset/Train/X_train.txt', sep=' ', header=None)
dataset_test = pd.read_csv('../data/HAPT_Dataset/Test/X_test.txt', sep=' ', header=None)


def get_data_generator():
    """Creates a data generator for the training.
    
    Args:
        time_series (bool): Indicates whether or not we want interpolated MNIST time series or just
            normal MNIST batches.
    
    Returns:
        generator: Data generator for the batches."""

    def batch_generator(mode="train", batch_size=100):
        """Generator for the data batches.
        
        Args:
            mode (str): Mode in ['train', 'val'] that decides which data set the generator
                samples from (default: 'train').
            batch_size (int): The size of the batches (default: 100).
            
        Yields:
            np.array: Data batch.
        """
        assert mode in ["train", "val"], "The mode should be in {train, val}."
        if mode=="train":
            images = data_train.copy()
            labels = labels_train.copy()
        elif mode=="val":
            images = data_val.copy()
            labels = labels_val.copy()
        
        while True:
            indices = np.random.permutation(np.arange(len(images)))
            images = images[indices]
            labels = labels[indices]

            for i in range(len(images)//batch_size):
                yield images[i*batch_size:(i+1)*batch_size]

    return batch_generator
# with open(data_set, 'rb') as som_dump:
#         _dataset = pickle.load(som_dump)
# dataset = _dataset['datasetNorm'].astype('float32')
# data = dataset.reshape(-1,288*3,1)


data_train = np.array(dataset_train)
labels_train = np.array(dataset_train)
data_val = np.array(dataset_test)
labels_val = np.array(dataset_test)
# numSamples = data.shape[0]
# numTrainSamples = int(numSamples*0.75)

# data_train = data[:numTrainSamples]
# labels_train = data[:numTrainSamples]
# data_val = data[numTrainSamples:numTrainSamples+10000]
# labels_val = data[numTrainSamples:numTrainSamples+10000]

input_length = data_train.shape[1]
input_channels = 1
latent_dim = 64
som_dim=[4,4]
encoder_hidden_size= 16
learning_rate = 0.0005
alpha = 1.0
beta = 0.9
gamma = 1.8
tau = 1.4
decay_factor = 0.9
batch_size =100

x = tf.placeholder(tf.float32, shape=[None, input_length, input_channels])

data_generator = get_data_generator()



# from keras.layers import Input, Dense
# from keras.models import Model
# encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# # with tf.device('/gpu:0'):
# input_img = tf.keras.Input(shape=(784,))
# encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_img)
# decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
# autoencoder = tf.keras.models.Model(input_img, decoded)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

with tf.Session() as sess:
    model = SOMAE(inputs=x, latent_dim=latent_dim, encoder_hidden_size=encoder_hidden_size, som_dim=som_dim, learning_rate=learning_rate, decay_factor=decay_factor,
                input_length=input_length, input_channels=input_channels, alpha=alpha, beta=beta, gamma=gamma,
                tau=tau)
    # print(model.encoder().summary())
    # print(model.decoder().summary())
    input_data = tf.keras.Input(shape=(input_length, input_channels, ), name='enc_input')
    h_1 = tf.keras.layers.LSTM(latent_dim, activation="relu", name='input2hid')(input_data)
    encoded_data = tf.keras.layers.Dense(encoder_hidden_size*latent_dim, activation="relu", name='hid2enc')(h_1)

    nearest_neuron_layer = tf.keras.layers.Lambda(model.winning_unit)(encoded_data)

    h_2 = tf.keras.layers.Dense(latent_dim, activation="relu",name='enc2hid')(nearest_neuron_layer)
    decoded = tf.keras.layers.Dense(input_length*input_channels, activation="linear", name='hid2dec')(h_2)
    autoencoder = tf.keras.models.Model(inputs=input_data, outputs=decoded)
print('OK')