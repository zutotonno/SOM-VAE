
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras


class SOMAE:
    def __init__(self, inputs, latent_dim=10, encoder_hidden_size = 64 , som_dim=[8,8], learning_rate=1e-4, decay_factor=0.95,
            input_length=288, input_channels=3, alpha=1., beta=1., gamma=1., tau=1.):
            self.inputs = inputs
            self.encoder_hidden_size = encoder_hidden_size
            self.latent_dim = latent_dim
            self.som_dim = som_dim
            self.learning_rate = learning_rate
            self.decay_factor = decay_factor
            self.input_length = input_length
            self.input_channels = input_channels
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.tau = tau

            self.inputs = keras.Input(shape=(input_length, input_channels, ), name='enc_input')
            h_1 = keras.layers.LSTM(self.latent_dim, activation="relu", name='input2hid')(self.inputs)
            encoded = keras.layers.Dense(self.encoder_hidden_size*self.latent_dim, activation="relu", name='hid2enc')(h_1)
            # encoded = tf.reshape(encoded,[-1, 1])
            # encoded = keras.layers
            self.encoder = keras.models.Model(inputs=self.inputs, outputs=encoded, name='encoder')
            print(self.encoder.summary())
            

            self.som = tf.Variable(tf.zeros([self.som_dim[0]*self.som_dim[1], self.latent_dim]))

            self.decoder_inputs = keras.Input(shape=(self.latent_dim, ))
            
            h_2 = keras.layers.Dense(self.latent_dim, activation="relu",name='enc2hid')(self.decoder_inputs)
            decoded = keras.layers.Dense(self.input_length*self.input_channels, activation="linear", name='hid2dec')(h_2)
            
            self.decoder = keras.models.Model(inputs=self.decoder_inputs, outputs=decoded, name="decoder")
            print(self.decoder.summary())


    def winning_unit(self, latent_representation):
        encoded_data_reshaped = tf.reshape(latent_representation,[-1, self.som_dim[0]*self.som_dim[1], self.latent_dim])
        squared_distances = tf.math.squared_difference(encoded_data_reshaped,self.som)
        component_sum = tf.reduce_sum(squared_distances, axis=-1)
        min_idx_reshaped = tf.argmin(component_sum, -1)
        # min_idx = tf.unravel_index(min_idx_reshaped, [self.som_dim[0],self.som_dim[1]])
        nearest_neuron = tf.gather(self.som,min_idx_reshaped)
        return nearest_neuron



    def compute_encodings(self):
        h_1 = tf.keras.layers.LSTM(self.latent_dim, activation="relu", name='input2hid')(self.inputs)
        encoded_data = tf.keras.layers.Dense(self.encoder_hidden_size*self.latent_dim, activation="relu", name='hid2enc')(h_1)
        return encoded_data
    
    def compute_embeddings(self):
    latent_representation = self.compute_encodings
    encoded_data_reshaped = tf.reshape(latent_representation,[-1, self.som_dim[0]*self.som_dim[1], self.latent_dim])
    squared_distances = tf.squared_difference(encoded_data_reshaped,self.som)
    component_sum = tf.reduce_sum(squared_distances, axis=-1)
    min_idx_reshaped = tf.argmin(component_sum, -1)
    # delta_w = tf.zeros([self.som_dim[0]*self.som_dim[1], self.latent_dim])
    # delta_w[min_idx_reshaped] = squared_distances*self.learning_rate
    nearest_neurons = tf.gather(self.som,min_idx_reshaped)
    # self.som = tf.add(self.som, delta_w)
    return nearest_neurons
    # min_idx_reshaped = tf.argmin(component_sum, -1)
    # # min_idx = tf.unravel_index(min_idx_reshaped, [self.som_dim[0],self.som_dim[1]])
    # nearest_neuron = tf.gather(self.som,min_idx_reshaped)
    # return nearest_neuron, min_idx_reshaped


    def loss_som(self):
        """Computes the SOM loss."""
        loss_som = tf.reduce_mean(tf.squared_difference((tf.stop_gradient(self.compute_encodings), self.compute_embeddings)))
        return loss_som

