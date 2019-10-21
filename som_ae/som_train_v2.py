import tensorflow as tf
from sklearn.datasets import make_blobs
from somTF_v2 import somTF
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import matplotlib
matplotlib.use('TkAgg')

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


data_train = np.reshape(x_train, [-1, 28*28])
np.random.shuffle(data_train)
data_val = np.reshape(x_test, [-1, 28*28])


def get_data_generator(train_data, validation_data):
    """Creates a data generator for the training."""

    def batch_generator(mode="train", batch_size=100):
        assert mode in ["train", "val"], "The mode should be in {train, val}."
        if mode == "train":
            data = train_data.copy()
        elif mode == "val":
            data = validation_data.copy()

        while True:
            indices = np.random.permutation(np.arange(len(data)))
            data = data[indices]

            for i in range(len(data) // batch_size):
                yield data[i * batch_size:(i + 1) * batch_size]

    return batch_generator


def train_model(model=None, x=None,  lr_val=None, num_epochs=None, patience=None, batch_size=None,
                learning_rate=None, interactive=True, generator=None, sess = None):
    """Trains the SOM-VAE model.

    Args:
        model (SOM-VAE): SOM-VAE model to train.
        x (tf.Tensor): Input tensor or placeholder.
        lr_val (tf.Tensor): Placeholder for the learning rate value.
        num_epochs (int): Number of epochs to train.
        patience (int): Patience parameter for the early stopping.
        batch_size (int): Batch size for the training generator.
        logdir (path): Directory for saving the logs.
        modelpath (path): Path for saving the model checkpoints.
        learning_rate (float): Learning rate for the optimization.
        interactive (bool): Indicator if we want to have an interactive
            progress bar for training.
        generator (generator): Generator for the data batches.
    """
    train_gen = generator("train", batch_size)
    val_gen = generator("val", batch_size)
    train_writer = tf.summary.FileWriter('../logs' + '/train', sess.graph)
    num_batches = len(data_train) // batch_size

    summaries = tf.compat.v1.summary.merge_all()
    sess.run(tf.compat.v1.global_variables_initializer())
    patience_count = 0
    test_losses = []
    print("Training...")
    train_step_SOMVAE = model.optimize
    if interactive:
        pbar = tqdm(total= num_epochs * num_batches)
    for epoch in range(num_epochs):
        batch_val = next(val_gen)
        test_loss, summary = sess.run([model.loss, summaries], feed_dict={x: batch_val})
        test_losses.append(test_loss)
        if test_losses[-1] == min(test_losses):
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= patience:
            break
        for i in range(num_batches):
            batch_data = next(train_gen)
            if i % 100 == 0:
                train_loss, summary, global_step = sess.run([model.loss, summaries, model.global_step], feed_dict={x: batch_data})
                train_writer.add_summary(summary, global_step=global_step)
            train_step_SOMVAE.run(feed_dict={x: batch_data, lr_val: learning_rate})
            if interactive:
                pbar.set_postfix(epoch=epoch, train_loss=train_loss, test_loss=test_loss, refresh=False)
                pbar.update(1)


if __name__ == '__main__':
    input_length = 28
    input_channels = 28
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
    data_generator = get_data_generator(data_train, data_val)
    som_dim = np.array([8, 8], dtype='int32')
    latent_dim = 28*28

    learning_rate = 0.0001
    alpha = 0.8
    beta = 0.9
    gamma = 1.8
    tau = 1.4
    decay_factor = 0.9
    decay_steps = 1000
    patience = 100
    num_epochs = 100
    batch_size = 1024
    lr_val = tf.compat.v1.placeholder_with_default(learning_rate, [])
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        model = somTF(inputs=x, latent_dim=latent_dim, som_dim=som_dim, learning_rate=lr_val,
                      decay_factor=decay_factor, decay_steps=decay_steps, batch_size=batch_size, alpha=alpha)
        train_model(model=model, x=x, lr_val=lr_val, learning_rate=learning_rate, interactive=True,
                    generator=data_generator, patience=patience, num_epochs=num_epochs, batch_size= batch_size,sess=sess)
        codebook = np.reshape(sess.run(model.codebook), [som_dim[0], som_dim[1],28,28])
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.6, wspace=0.1)
        k = 0
        for i, j in itertools.product(range(som_dim[0]), range(som_dim[1])):
            ax = fig.add_subplot(som_dim[0], som_dim[1], k + 1)
            ax.matshow(codebook[i,j])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            k += 1
        plt.show()
        print('Finish')
    print('Closing session')




