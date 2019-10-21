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
            with tf.compat.v1.variable_scope(function.__name__):
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
        self.BMU
        self.neighbors
        self.loss_bmu
        self.loss_neigh
        self.loss
        self.optimize

    @lazy_scope
    def codebook(self):
        """Creates variable for the SOM embeddings."""
        embeddings = tf.compat.v1.get_variable("codebook", [self.som_dim[0]*self.som_dim[1] , self.latent_dim],
                                     initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.001))
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
        return min_idx

    @lazy_scope
    def BMU(self):
        min_idx = self.k_index
        # k_1 = min_idx // self.som_dim[0]
        # k_2 = min_idx % self.som_dim[1]
        # k_stacked = tf.stack([k_1, k_2], axis=1)
        # codebook_reshaped = tf.reshape(self.codebook,[self.som_dim[0], self.som_dim[1], self.latent_dim])
        nearest_neuron = tf.gather(params=tf.expand_dims(self.codebook, 0), indices=min_idx)
        return nearest_neuron


    def ravel_multi_index(x):
        ''' x is a two dimensional index'''



    @lazy_scope
    def neighbors(self):
        min_idx = self.k_index
        k_1 = min_idx // self.som_dim[0]
        k_2 = min_idx % self.som_dim[1]

        k1_not_top = tf.less(k_1, tf.constant(self.som_dim[0] - 1, dtype=tf.int64))
        k1_not_bottom = tf.greater(k_1, tf.constant(0, dtype=tf.int64))
        k2_not_right = tf.less(k_2, tf.constant(self.som_dim[1] - 1, dtype=tf.int64))
        k2_not_left = tf.greater(k_2, tf.constant(0, dtype=tf.int64))

        k1_up = tf.compat.v1.where(k1_not_top, tf.add(k_1, 1), k_1)
        k1_down = tf.compat.v1.where(k1_not_bottom, tf.subtract(k_1, 1), k_1)
        k2_right = tf.compat.v1.where(k2_not_right, tf.add(k_2, 1), k_2)
        k2_left = tf.compat.v1.where(k2_not_left, tf.subtract(k_2, 1), k_2)

        nearest_neuron = tf.gather(params=tf.expand_dims(self.codebook,0), indices=min_idx)

        up_nearest_neuron = tf.compat.v1.where(k1_not_top, tf.gather_nd(tf.expand_dims(self.codebook, 0), tf.stack([k1_up, k_2], axis=1)),
                                     tf.zeros([self.batch_size, self.latent_dim]))
        bottom_nearest_neuron = tf.compat.v1.where(k1_not_bottom, tf.gather_nd(tf.expand_dims(self.codebook, 0), tf.stack([k1_down, k_2], axis=1)),
                            tf.zeros([self.batch_size, self.latent_dim]))
        right_nearest_neuron = tf.compat.v1.where(k2_not_right, tf.gather_nd(tf.expand_dims(self.codebook, 0), tf.stack([k_1, k2_right], axis=1)),
                             tf.zeros([self.batch_size, self.latent_dim]))
        left_nearest_neuron = tf.compat.v1.where(k2_not_left, tf.gather_nd(tf.expand_dims(self.codebook, 0), tf.stack([k_1, k2_left], axis=1)),
                            tf.zeros([self.batch_size, self.latent_dim]))

        som_neighbors = tf.stack([nearest_neuron, up_nearest_neuron, bottom_nearest_neuron, left_nearest_neuron,
                                  right_nearest_neuron], axis=1)
        tf.compat.v1.summary.histogram("BMU_O", k_1)
        tf.compat.v1.summary.histogram("BMU_1", k_2)
        return som_neighbors

    @lazy_scope
    def update_neighbors(self, lr, alpha):
        min_idx = self.k_index
        k_1 = min_idx // self.som_dim[0]
        k_2 = min_idx % self.som_dim[1]

        k1_not_top = tf.less(k_1, tf.constant(self.som_dim[0] - 1, dtype=tf.int64))
        k1_not_bottom = tf.greater(k_1, tf.constant(0, dtype=tf.int64))
        k2_not_right = tf.less(k_2, tf.constant(self.som_dim[1] - 1, dtype=tf.int64))
        k2_not_left = tf.greater(k_2, tf.constant(0, dtype=tf.int64))

        k1_up = tf.compat.v1.where(k1_not_top, tf.add(k_1, 1), k_1)
        k1_down = tf.compat.v1.where(k1_not_bottom, tf.subtract(k_1, 1), k_1)
        k2_right = tf.compat.v1.where(k2_not_right, tf.add(k_2, 1), k_2)
        k2_left = tf.compat.v1.where(k2_not_left, tf.subtract(k_2, 1), k_2)
        up_nearest_neuron = tf.compat.v1.where(k1_not_top, tf.gather_nd(tf.expand_dims(self.codebook, 0),
                                                                        tf.stack([k1_up, k_2], axis=1)),
                                               tf.zeros([self.batch_size, self.latent_dim]))
        bottom_nearest_neuron = tf.compat.v1.where(k1_not_bottom, tf.gather_nd(tf.expand_dims(self.codebook, 0),
                                                                               tf.stack([k1_down, k_2], axis=1)),
                                                   tf.zeros([self.batch_size, self.latent_dim]))
        right_nearest_neuron = tf.compat.v1.where(k2_not_right, tf.gather_nd(tf.expand_dims(self.codebook, 0),
                                                                             tf.stack([k_1, k2_right], axis=1)),
                                                  tf.zeros([self.batch_size, self.latent_dim]))
        left_nearest_neuron = tf.compat.v1.where(k2_not_left, tf.gather_nd(tf.expand_dims(self.codebook, 0),
                                                                           tf.stack([k_1, k2_left], axis=1)),
                                                 tf.zeros([self.batch_size, self.latent_dim]))

        up_nearest_delta = tf.reduce_mean(tf.math.squared_difference(up_nearest_neuron, self.inputs), axis=1)*lr*alpha
        bottom_nearest_delta = tf.reduce_mean(tf.math.squared_difference(bottom_nearest_neuron, self.inputs),
                                              axis=1) * lr * alpha
        right_nearest_delta = tf.reduce_mean(tf.math.squared_difference(right_nearest_neuron, self.inputs),
                                              axis=1) * lr * alpha
        left_nearest_delta = tf.reduce_mean(tf.math.squared_difference(left_nearest_neuron, self.inputs),
                                             axis=1) * lr * alpha
        # self.codebook[k1_down, k_2] = tf.add()


    @lazy_scope
    def loss_neigh(self):
        loss_som = tf.reduce_mean(
            input_tensor=tf.math.squared_difference(tf.expand_dims(self.inputs, axis=1), self.neighbors))
        tf.compat.v1.summary.scalar("loss_neigh", loss_som)
        return loss_som

    @lazy_scope
    def loss_bmu(self):
        loss_som = tf.reduce_mean(
            input_tensor=tf.math.squared_difference(self.inputs, self.BMU))
        tf.compat.v1.summary.scalar("loss_bmu", loss_som)
        return loss_som

    @lazy_scope
    def loss(self):
        """Aggregates the loss terms into the total loss."""
        loss = (self.loss_bmu + self.alpha * self.loss_neigh)
        tf.summary.scalar("loss", loss)
        return loss


    @lazy_scope
    def optimize(self):
        """Optimizes the model's loss using Adam with exponential learning rate decay."""
        lr_decay = tf.compat.v1.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_factor, staircase=True)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_decay)
        train_step = optimizer.minimize(self.loss, global_step=self.global_step)
        return train_step
