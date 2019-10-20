
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
        squared_distances = tf.math.squared_difference(encoded_data_reshaped, self.som)
        component_sum = tf.reduce_sum(squared_distances, axis=-1)
        min_idx_reshaped = tf.argmin(component_sum, -1)
        # min_idx = tf.unravel_index(min_idx_reshaped, [self.som_dim[0],self.som_dim[1]])
        nearest_neuron = tf.gather(self.som,min_idx_reshaped)
        return nearest_neuron


    def compute_encodings(self):
        h_1 = tf.keras.layers.LSTM(self.latent_dim, activation="relu", name='input2hid')(self.inputs)
        encoded_data = tf.keras.layers.Dense(self.encoder_hidden_size*self.latent_dim, activation="relu", name='hid2enc')(h_1)
        return encoded_data


    def compute_BMU(self):
        latent_representation = self.compute_encodings
        encoded_data_reshaped = tf.reshape(latent_representation,
                                           [-1, self.som_dim[0] * self.som_dim[1], self.latent_dim])
        squared_distances = tf.squared_difference(encoded_data_reshaped, self.som)
        component_sum = tf.reduce_sum(squared_distances, axis=-1)
        min_idx = tf.argmin(component_sum, -1)
        return min_idx

    def compute_embeddings(self):
        min_idx = self.compute_BMU
        nearest_neuron = tf.gather(self.som, min_idx)
        return nearest_neuron

    def compute_neighbors_embeddings(self):
        min_idx = self.compute_BMU
        min_idx_reshaped = tf.unravel_index(min_idx, self.som_dim[0], self.som_dim[1])
        up_idx = tf.cond(min_idx_reshaped[0] > tf.constant(0), lambda: tf.subtract(min_idx_reshaped[0], 1), lambda: min_idx_reshaped[0])
        bottom_idx = tf.cond(min_idx_reshaped[0] < self.som_dim[0], lambda: tf.add(min_idx_reshaped[0], 1), lambda: min_idx_reshaped[0])
        left_idx = tf.cond(min_idx_reshaped[1] > tf.constant(0), lambda: tf.subtract(min_idx_reshaped[1], 1), lambda: min_idx_reshaped[1])
        right_idx = tf.cond(min_idx_reshaped[1] < tf.constant(0), lambda: tf.add(min_idx_reshaped[1], 1), lambda: min_idx_reshaped[1])

        up_idx_reshaped = tf.unravel_index(up_idx, self.som_dim[0], self.som_dim[1])
        bottom_idx_reshaped = tf.unravel_index(bottom_idx, self.som_dim[0], self.som_dim[1])
        left_idx_reshaped = tf.unravel_index(left_idx, self.som_dim[0], self.som_dim[1])
        right_idx_reshaped = tf.unravel_index(right_idx, self.som_dim[0], self.som_dim[1])
        som_reshaped = tf.reshape(self.som, [-1, self.som_dim[0], self.som_dim[1], self.latent_dim])

        nearest_neuron = tf.gather_nd(som_reshaped, min_idx_reshaped)
        up_nearest_neuron = tf.gather(som_reshaped, up_idx_reshaped)
        bottom_nearest_neuron = tf.gather(som_reshaped, bottom_idx_reshaped)
        left_nearest_neuron = tf.gather(som_reshaped, left_idx_reshaped)
        right_nearest_neuron = tf.gather(som_reshaped, right_idx_reshaped)

        som_neighbors = tf.stack([nearest_neuron, up_nearest_neuron, bottom_nearest_neuron, left_nearest_neuron,
                                  right_nearest_neuron], axis=1)
        return som_neighbors

    # def update_neighbors(self, lr, area):
    #     latent_representation = self.compute_encodings
    #     encoded_data_reshaped = tf.reshape(latent_representation,
    #                                        [-1, self.som_dim[0] * self.som_dim[1], self.latent_dim])
    #     squared_distances = tf.squared_difference(encoded_data_reshaped, self.som)
    #     component_sum = tf.reduce_sum(squared_distances, axis=-1)
    #     min_idx_reshaped = tf.argmin(component_sum, -1)

    def loss_commit(self):
        """ This loss term is minimized in order to move the embedding to the direction of the encodings"""
        loss_commit = tf.reduce_mean(tf.squared_difference(self.compute_encodings,
                                                           self.compute_embeddings))
        tf.summary.scalar("loss_commit", loss_commit)
        return loss_commit

    def loss_som(self):
        """Computes the SOM loss."""
        loss_som = tf.reduce_mean(tf.squared_difference((tf.stop_gradient(self.compute_encodings),
                                                         self.compute_neighbors_embeddings)))
        tf.summary.scalar("loss", loss_som)
        return loss_som


    def optimize(self):
        """Optimizes the model's loss using Adam with exponential learning rate decay."""
        lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                              self.decay_factor, staircase=True)
        neigh_area_decay = tf.train.exponential_decay(self.neigh_area, self.global_step, self.decay_steps,
                                              self.decay_factor, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr_decay)
        train_step = optimizer.minimize(self.loss, global_step=self.global_step)
        # som_train_step = self.update_neighbors(learning_rate=lr_decay, neigh_area=neigh_area_decay)
        return train_step
