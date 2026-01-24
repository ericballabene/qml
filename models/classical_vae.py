import tensorflow as tf
import os
import numpy as np

class ClassicalVAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, model_name="ClassicalVAE", save_dir="ClassicalVAE"):
        super().__init__(name=model_name)
        self.latent_dim = latent_dim
        self.save_dir = save_dir

        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.LeakyReLU(alpha=0.2),  # alpha can be 0.01 ~ 0.3
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.LeakyReLU(alpha=0.2),  # alpha can be 0.01 ~ 0.3
            tf.keras.layers.Dense(latent_dim * 2),
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.LeakyReLU(alpha=0.2),  # alpha can be 0.01 ~ 0.3
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.LeakyReLU(alpha=0.2),  # alpha can be 0.01 ~ 0.3
            tf.keras.layers.Dense(input_dim),
        ])

    def sample(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean), dtype=tf.float32)
        return mean + tf.exp(0.5 * log_var) * eps

    def call(self, x):
        mean_log_var = self.encoder(x)
        mean, log_var = tf.split(mean_log_var, 2, axis=1)
        z = self.sample(mean, log_var)
        x_hat = self.decoder(z)
        # ensure float32
        return tf.cast(x_hat, tf.float32), tf.cast(mean, tf.float32), tf.cast(log_var, tf.float32)


def train_ClassicalVAE(model, dataset, epochs=50, learning_rate=1e-4, model_name='UNK', save_dir='UNK'):
    """Train VAE with stability fixes + save loss history (DNN-like)."""

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    history_loss = []
    history_reco = []
    history_kl = []

    for epoch in range(epochs):
        epoch_reco = 0.0
        epoch_kl = 0.0
        epoch_total = 0.0
        num_batches = 0

        for batch in dataset:
            with tf.GradientTape() as tape:
                x_hat, mean, log_var = model(batch)
                log_var = tf.clip_by_value(log_var, -10.0, 10.0)

                reco_loss = mse_loss_fn(batch, x_hat)
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + log_var - tf.square(mean) - tf.exp(log_var)
                )
                loss = reco_loss + kl_loss

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_reco += reco_loss.numpy()
            epoch_kl += kl_loss.numpy()
            epoch_total += loss.numpy()
            num_batches += 1

        epoch_reco /= num_batches
        epoch_kl /= num_batches
        epoch_total /= num_batches

        history_reco.append(epoch_reco)
        history_kl.append(epoch_kl)
        history_loss.append(epoch_total)

        print(
            f"[VAE {model_name}] Epoch {epoch+1}/{epochs} | "
            f"Reco: {epoch_reco:.6f} | "
            f"KL: {epoch_kl:.6f} | "
            f"Total: {epoch_total:.6f}"
        )

    # --- save loss curves like DNN ---
    np.savez(
        f"{save_dir}/VAE_{model_name}.npz",
        loss=np.array(history_loss),
        reco_loss=np.array(history_reco),
        kl_loss=np.array(history_kl),
    )

    print(f"[VAE {model_name}] Saved training history to VAE_{model_name}.npz")


def train_on_subset(X_subset, latent_dim=10, epochs=50, model_name="A", save_dir="output_vae"):
    """
    Train a ClassicalVAE on a dataset subset.
    If weights already exist, load them instead of retraining.
    """
    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, f"classical_vae_{model_name}.h5")
    vae_model = ClassicalVAE(input_dim=X_subset.shape[1], latent_dim=latent_dim, model_name=model_name, save_dir=save_dir)

    if os.path.exists(weight_path):
        # Build variables by calling model once
        dummy_input = tf.convert_to_tensor(X_subset[:1], dtype=tf.float32)
        vae_model(dummy_input)
        vae_model.load_weights(weight_path)
        print(f"Model weights already exist. Loaded: {weight_path}")
        return vae_model

    # TF dataset
    dataset = tf.data.Dataset.from_tensor_slices(X_subset.astype(np.float32))
    dataset = dataset.shuffle(buffer_size=len(X_subset), seed=47).batch(128)

    print(f"Training ClassicalVAE model {model_name} on {len(X_subset)} samples...")
    train_ClassicalVAE(vae_model, dataset, epochs=epochs, learning_rate=1e-4, model_name=model_name, save_dir=save_dir)

    vae_model.save_weights(weight_path)
    print(f"Model weights saved: {weight_path}")
    return vae_model

