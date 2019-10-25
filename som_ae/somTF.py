import tensorflow as tf
import functools
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import io
# import itertools
# import numpy as np


def lazy_scope(function):
    """Creates a decorator for methods that makes their return values load lazily.

    A method with this decorator will only compute the return value once when called
    for the first time. Afterwards, the value will be cached as an object attribute.
    Inspired by: https://danijar.com/structuring-your-tensorflow-models

    Args:
        function (func): Function to be decorated.

    Returns:
        decorator: Decorator for the function.
    """
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class somTF:
    def __init__(self, inputs=None, latent_dim=None , som_dim=None, learning_rate=None,
                 decay_factor=None, decay_steps=None, batch_size = None, alpha = None):
        self.inputs = inputs
        self.som_dim = som_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        self.alpha = alpha

        self.codebook
        self.global_step
        self.k_index
        # self.loss_bmu
        # self.loss_neigh
        self.loss_bmu
        self.loss_bmu_up
        self.loss_bmu_bottom
        self.loss_bmu_right
        self.loss_bmu_left
        self.loss
        self.optimize

        # self.plot_neurons

    @lazy_scope
    def codebook(self):
        """Creates variable for the SOM embeddings."""
        embeddings = tf.get_variable("codebook", [self.som_dim[0]*self.som_dim[1] , self.latent_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.05))
        tf.compat.v1.summary.tensor_summary("codebook", embeddings)
        return embeddings

    @lazy_scope
    def global_step(self):
        """Creates global_step variable for the optimization."""
        global_step = tf.Variable(0, trainable=False, name="global_step")
        return global_step

    @lazy_scope
    def k_index(self):
        squared_distances = tf.math.squared_difference(tf.expand_dims(self.inputs, 1),
                                                  tf.expand_dims(self.codebook, 0))
        component_sum = tf.reduce_sum(input_tensor=squared_distances, axis=-1)
        min_idx = tf.argmin(input=component_sum, axis=-1)
        k_0 = min_idx // self.som_dim[0]
        k_1 = min_idx % self.som_dim[1]
        tf.compat.v1.summary.histogram("bmu_0", k_0)
        tf.compat.v1.summary.histogram("bmu_1", k_1)
        return min_idx

    @lazy_scope
    def BMU(self):
        min_idx = self.k_index
        nearest_neuron = tf.gather(params=self.codebook, indices=min_idx)
        return nearest_neuron

    # @lazy_scope
    # def neighbors(self):
    #     min_idx = self.k_index
    #     nearest_neuron = tf.gather(params=self.codebook, indices=min_idx)
    #     up_nearest_neuron = tf.gather(params=self.codebook,  indices=self.select_up)
    #     bottom_nearest_neuron = tf.gather(params=self.codebook,  indices=self.select_bottom)
    #     left_nearest_neuron = tf.gather(params=self.codebook,  indices=self.select_left)
    #     right_nearest_neuron = tf.gather(params=self.codebook,  indices=self.select_right)
    #     # k_1 = min_idx // self.som_dim[0]
    #     # k_2 = min_idx % self.som_dim[1]
    #     # k_stacked = tf.stack([k_1, k_2], axis=1)

    #     # k1_not_top = tf.less(k_1, tf.constant(self.som_dim[0] - 1, dtype=tf.int64))
    #     # k1_not_bottom = tf.greater(k_1, tf.constant(0, dtype=tf.int64))
    #     # k2_not_right = tf.less(k_2, tf.constant(self.som_dim[1] - 1, dtype=tf.int64))
    #     # k2_not_left = tf.greater(k_2, tf.constant(0, dtype=tf.int64))

    #     # k1_up = tf.where(k1_not_top, tf.add(k_1, 1), k_1)
    #     # k1_down = tf.where(k1_not_bottom, tf.subtract(k_1, 1), k_1)
    #     # k2_right = tf.where(k2_not_right, tf.add(k_2, 1), k_2)
    #     # k2_left = tf.where(k2_not_left, tf.subtract(k_2, 1), k_2)

    #     # nearest_neuron = tf.gather_nd(params=self.codebook, indices=k_stacked, batch_dims=self.batch_size)

    #     # up_nearest_neuron = tf.where(k1_not_top, tf.gather_nd(self.codebook, tf.stack([k1_up, k_2], axis=1),
    #     #                                                       batch_dims=self.batch_size),
    #     #                              tf.zeros([self.batch_size, self.latent_dim]))
    #     # bottom_nearest_neuron = tf.where(k1_not_bottom, tf.gather_nd(self.codebook, tf.stack([k1_down, k_2], axis=1),
    #     #                                                              batch_dims=self.batch_size),
    #     #                     tf.zeros([self.batch_size, self.latent_dim]))
    #     # right_nearest_neuron = tf.where(k2_not_right, tf.gather_nd(self.codebook, tf.stack([k_1, k2_right], axis=1),
    #     #                                                            batch_dims=self.batch_size),
    #     #                      tf.zeros([self.batch_size, self.latent_dim]))
    #     # left_nearest_neuron = tf.where(k2_not_left, tf.gather_nd(self.codebook, tf.stack([k_1, k2_left], axis=1),
    #     #                                                          batch_dims=self.batch_size),
    #     #                     tf.zeros([self.batch_size, self.latent_dim]))

    #     som_neighbors = tf.stack([nearest_neuron, up_nearest_neuron, bottom_nearest_neuron, left_nearest_neuron,
    #                               right_nearest_neuron], axis=1)
    #     return som_neighbors

    @lazy_scope
    def loss_bmu(self):
        loss_som = tf.reduce_mean(
            tf.squared_difference(self.inputs, self.BMU))
        tf.compat.v1.summary.scalar("bmu_loss", loss_som)
        return loss_som

    @lazy_scope
    def loss_bmu_up(self):
        bmu = self.k_index
        _bmu = tf.unravel_index(bmu,[self.som_dim[0], self.som_dim[1]])
        zeros = tf.zeros_like(_bmu[0])
        mask = tf.greater(_bmu[0], zeros) # boolean tensor, mask[i] = True iff x[i] > 0
        _bmu_movable_sx = tf.boolean_mask(_bmu[0], mask)
        _bmu_movable_dx = tf.boolean_mask(_bmu[1], mask)
        _bmu_raveled = tf.multiply(tf.add(_bmu_movable_sx, -1),self.som_dim[0]) + (_bmu_movable_dx % self.som_dim[1])
        _inputs_idx = tf.boolean_mask(self.inputs, mask)
        current_loss = tf.reduce_mean(
            tf.squared_difference(_inputs_idx, tf.gather(self.codebook, _bmu_raveled)))
        return current_loss
        # return _bmu_raveled


    @lazy_scope
    def loss_bmu_bottom(self):
        bmu = self.k_index
        _bmu = tf.unravel_index(bmu,[self.som_dim[0], self.som_dim[1]])
        bottomers = tf.multiply(tf.ones_like(_bmu[0]), self.som_dim[1])
        mask = tf.less(_bmu[0], bottomers) # boolean tensor, mask[i] = True iff x[i] > 0
        _bmu_movable_sx = tf.boolean_mask(_bmu[0], mask)
        _bmu_movable_dx = tf.boolean_mask(_bmu[1], mask)
        _bmu_raveled = tf.multiply(tf.add(_bmu_movable_sx, 1),self.som_dim[0]) + (_bmu_movable_dx % self.som_dim[1])
        _inputs_idx = tf.boolean_mask(self.inputs, mask)
        current_loss = tf.reduce_mean(
            tf.squared_difference(_inputs_idx, tf.gather(self.codebook, _bmu_raveled)))
        return current_loss

    @lazy_scope
    def loss_bmu_left(self):
        bmu = self.k_index
        _bmu = tf.unravel_index(bmu,[self.som_dim[0], self.som_dim[1]])
        zeros = tf.zeros_like(_bmu[1])
        mask = tf.greater(_bmu[1], zeros) # boolean tensor, mask[i] = True iff x[i] > 0
        _bmu_movable_sx = tf.boolean_mask(_bmu[0], mask)
        _bmu_movable_dx = tf.boolean_mask(_bmu[1], mask)
        _bmu_raveled = tf.multiply(_bmu_movable_sx,self.som_dim[0]) + (tf.add(_bmu_movable_dx, -1) % self.som_dim[1])
        _inputs_idx = tf.boolean_mask(self.inputs, mask)
        current_loss = tf.reduce_mean(
            tf.squared_difference(_inputs_idx, tf.gather(self.codebook, _bmu_raveled)))
        return current_loss


    @lazy_scope
    def loss_bmu_right(self):
        bmu = self.k_index
        _bmu = tf.unravel_index(bmu,[self.som_dim[0], self.som_dim[1]])
        righters = tf.multiply(tf.ones_like(_bmu[1]), self.som_dim[1])
        mask = tf.less(_bmu[1], righters) # boolean tensor, mask[i] = True iff x[i] > 0
        _bmu_movable_sx = tf.boolean_mask(_bmu[0], mask)
        _bmu_movable_dx = tf.boolean_mask(_bmu[1], mask)
        _bmu_raveled = tf.multiply(_bmu_movable_sx,self.som_dim[0]) + (tf.add(_bmu_movable_dx, 1) % self.som_dim[1])
        _inputs_idx = tf.boolean_mask(self.inputs, mask)
        current_loss = tf.reduce_mean(
            tf.squared_difference(_inputs_idx, tf.gather(self.codebook, _bmu_raveled)))
        return current_loss

    @lazy_scope
    def loss(self):
        loss = (self.loss_bmu + self.alpha*self.loss_bmu_up + self.alpha*self.loss_bmu_bottom
                + self.alpha*self.loss_bmu_left + self.alpha*self.loss_bmu_right )
        tf.compat.v1.summary.scalar("loss", loss)
        return loss

    @lazy_scope
    def optimize(self):
        """Optimizes the model's loss using Adam with exponential learning rate decay."""
        lr_decay = tf.compat.v1.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_factor, staircase=True)
        optimizer = tf.compat.v1.train.AdamOptimizer(lr_decay)
        train_step = optimizer.minimize(self.loss, global_step=self.global_step)
        return train_step

    # @lazy_scope
    # def plot_neurons(self):
    #     codebook = np.reshape(self.codebook.eval(), [self.som_dim[0], self.som_dim[1],28,28])
    #     fig = plt.figure(figsize=(20,20))
    #     fig.subplots_adjust(hspace=0.2, wspace=0.05)
    #     k = 0
    #     for i, j in itertools.product(range(self.som_dim[0]), range(self.som_dim[1])):
    #         ax = fig.add_subplot(self.som_dim[0], self.som_dim[1], k + 1)
    #         ax.matshow(codebook[i,j])
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #         k += 1
    #     plt.show()
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='png')
    #     plt.savefig('neurons.png')
    #     # Closing the figure prevents it from being displayed directly inside
    #     # the notebook.
    #     plt.close(fig)
    #     buf.seek(0)
    #     # Convert PNG buffer to TF image
    #     image = tf.image.decode_png(buf.getvalue(), channels=1)
    #     # Add the batch dimension
    #     image = tf.expand_dims(image, 0)
    #     tf.compat.v1.summary.image("SOM Neurons", image)