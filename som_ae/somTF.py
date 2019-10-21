import tensorflow as tf
import functools

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
                 decay_factor=None, decay_steps=None, batch_size = None):
        self.inputs = inputs
        self.som_dim = som_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.batch_size = batch_size

        self.codebook
        self.global_step
        self.BMU
        self.loss
        self.optimize

    @lazy_scope
    def codebook(self):
        """Creates variable for the SOM embeddings."""
        embeddings = tf.get_variable("codebook", [self.som_dim[0]*self.som_dim[1] , self.latent_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.05))
        tf.summary.tensor_summary("codebook", embeddings)
        return embeddings

    @lazy_scope
    def global_step(self):
        """Creates global_step variable for the optimization."""
        global_step = tf.Variable(0, trainable=False, name="global_step")
        return global_step

    @lazy_scope
    def BMU(self):
        squared_distances = tf.squared_difference(tf.expand_dims(self.inputs, 1),
                                                  tf.expand_dims(self.codebook, 0))
        component_sum = tf.reduce_sum(squared_distances, axis=-1)
        min_idx = tf.argmin(component_sum, -1)
        return min_idx

    @lazy_scope
    def neighbors(self):
        min_idx = self.BMU
        # min_idx_reshaped_0 = min_idx // self.som_dim[0]
        # min_idx_reshaped_1 = min_idx % self.som_dim[1]
        # min_idx_reshaped = tf.stack([min_idx_reshaped_0, min_idx_reshaped_1], axis=1)
        # min_idx_reshaped = tf.dtypes.cast(min_idx_reshaped, tf.int32)
        # up_idx = tf.cond(min_idx_reshaped[0] > tf.constant(0), lambda: tf.add(min_idx_reshaped, tf.constant([-1, 0])),
        #                  lambda: min_idx_reshaped)
        # bottom_idx = tf.cond(min_idx_reshaped[0] < self.som_dim[0],
        #                      lambda: tf.add(min_idx_reshaped, tf.constant([1, 0])),
        #                      lambda: min_idx_reshaped)
        # left_idx = tf.cond(min_idx_reshaped[1] > tf.constant(0), lambda: tf.add(min_idx_reshaped, tf.constant([0, -1])),
        #                    lambda: min_idx_reshaped)
        # right_idx = tf.cond(min_idx_reshaped[1] < self.som_dim[0],
        #                     lambda: tf.add(min_idx_reshaped, tf.constant([0, 1])),
        #                     lambda: min_idx_reshaped)
        #
        # som_reshaped = tf.reshape(self.codebook, [self.som_dim[0], self.som_dim[1], self.latent_dim])
        #
        # nearest_neuron = tf.gather_nd(som_reshaped, min_idx_reshaped)
        # up_nearest_neuron = tf.gather_nd(som_reshaped, up_idx)
        # bottom_nearest_neuron = tf.gather_nd(som_reshaped, bottom_idx)
        # left_nearest_neuron = tf.gather_nd(som_reshaped, left_idx)
        # right_nearest_neuron = tf.gather_nd(som_reshaped, right_idx)

        k_1 = min_idx // self.som_dim[0]
        k_2 = min_idx % self.som_dim[1]
        k_stacked = tf.stack([k_1, k_2], axis=1)

        k1_not_top = tf.less(k_1, tf.constant(self.som_dim[0] - 1, dtype=tf.int64))
        k1_not_bottom = tf.greater(k_1, tf.constant(0, dtype=tf.int64))
        k2_not_right = tf.less(k_2, tf.constant(self.som_dim[1] - 1, dtype=tf.int64))
        k2_not_left = tf.greater(k_2, tf.constant(0, dtype=tf.int64))

        k1_up = tf.where(k1_not_top, tf.add(k_1, 1), k_1)
        k1_down = tf.where(k1_not_bottom, tf.subtract(k_1, 1), k_1)
        k2_right = tf.where(k2_not_right, tf.add(k_2, 1), k_2)
        k2_left = tf.where(k2_not_left, tf.subtract(k_2, 1), k_2)

        nearest_neuron = tf.gather_nd(params=self.codebook, indices=k_stacked, batch_dims=self.batch_size)

        up_nearest_neuron = tf.where(k1_not_top, tf.gather_nd(self.codebook, tf.stack([k1_up, k_2], axis=1),
                                                              batch_dims=self.batch_size),
                                     tf.zeros([self.batch_size, self.latent_dim]))
        bottom_nearest_neuron = tf.where(k1_not_bottom, tf.gather_nd(self.codebook, tf.stack([k1_down, k_2], axis=1),
                                                                     batch_dims=self.batch_size),
                            tf.zeros([self.batch_size, self.latent_dim]))
        right_nearest_neuron = tf.where(k2_not_right, tf.gather_nd(self.codebook, tf.stack([k_1, k2_right], axis=1),
                                                                   batch_dims=self.batch_size),
                             tf.zeros([self.batch_size, self.latent_dim]))
        left_nearest_neuron = tf.where(k2_not_left, tf.gather_nd(self.codebook, tf.stack([k_1, k2_left], axis=1),
                                                                 batch_dims=self.batch_size),
                            tf.zeros([self.batch_size, self.latent_dim]))

        som_neighbors = tf.stack([nearest_neuron, up_nearest_neuron, bottom_nearest_neuron, left_nearest_neuron,
                                  right_nearest_neuron], axis=1)
        return som_neighbors

    @lazy_scope
    def loss(self):
        loss_som = tf.reduce_mean(
            tf.squared_difference(tf.expand_dims(tf.stop_gradient(self.inputs), axis=2), self.neighbors))
        tf.summary.scalar("loss_som", loss_som)
        return loss_som

    @lazy_scope
    def optimize(self):
        """Optimizes the model's loss using Adam with exponential learning rate decay."""
        lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_factor, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr_decay)
        train_step = optimizer.minimize(self.loss, global_step=self.global_step)
        return train_step
