import tensorflow as tf
from sklearn.datasets import make_blobs
from somTF import somTF
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm


mnist = input_data.read_data_sets(f"../data/MNIST_data")

data_train = np.reshape(mnist.train.images, [-1, 28*28])
data_val = data_train[45000:]
data_train = data_train[:45000]


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
                learning_rate=None, interactive=True, generator=None):
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

    num_batches = len(data_train) // batch_size

    summaries = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
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
                    train_loss, summary = sess.run([model.loss, summaries], feed_dict={x: batch_data})
                train_step_SOMVAE.run(feed_dict={x: batch_data, lr_val: learning_rate})
                if interactive:
                    pbar.set_postfix(epoch=epoch, train_loss=train_loss, test_loss=test_loss, refresh=False)
                    pbar.update(1)
            if interactive:
                pbar.close()


if __name__ == '__main__':
    input_length = 28
    input_channels = 28
    x = tf.placeholder(tf.float32, shape=[None, 28*28])
    learning_rate = 0.0005
    data_generator = get_data_generator(data_train, data_val)
    som_dim = np.array([4, 4], dtype='int32')
    latent_dim = 28*28
    alpha = 1.0
    beta = 0.9
    gamma = 1.8
    tau = 1.4
    decay_factor = 0.9
    decay_steps = 1000
    patience = 100
    num_epochs = 20
    batch_size = 128
    lr_val = tf.placeholder_with_default(learning_rate, [])


    model = somTF(inputs=x, latent_dim=latent_dim, som_dim=som_dim, learning_rate=lr_val,
                  decay_factor=decay_factor, decay_steps=decay_steps, batch_size=batch_size)
    train_model(model=model, x=x, lr_val=lr_val, learning_rate=learning_rate, interactive=True,
                generator=data_generator, patience=patience, num_epochs=num_epochs, batch_size= batch_size)
    codebook = tf.reshape(model.codebook, [28, 28])
    print('Finish')




