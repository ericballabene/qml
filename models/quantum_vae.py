import os
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy


# ============================================================
# Quantum circuit builder
# ============================================================
def build_latent_circuit(latent_dim):
    """Builds a parameterized quantum circuit for the latent space."""
    qubits = cirq.GridQubit.rect(1, latent_dim)

    # Symbols for data and trainable parameters
    x_symbols = sympy.symbols(f'x0:{latent_dim}')
    theta_symbols = sympy.symbols(f'theta0:{latent_dim}')

    circuit = cirq.Circuit()

    # Encode classical data (mean) into rotations
    for i, q in enumerate(qubits):
        circuit.append(cirq.ry(x_symbols[i])(q))

    # Variational layer
    for i, q in enumerate(qubits):
        circuit.append(cirq.ry(theta_symbols[i])(q))

    # Simple entanglement
    for i in range(latent_dim - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    return circuit, qubits, x_symbols, theta_symbols

def create_reuploading_pqc_with_multiqubit_correlators(
    num_qubits,
    num_layers=3,
):
    qubits = cirq.GridQubit.rect(1, num_qubits)

    data_symbols = sympy.symbols(f'x0:{num_qubits}')
    theta_symbols = sympy.symbols(
        f'theta0:{num_layers * num_qubits}'
    )

    circuit = cirq.Circuit()
    theta_idx = 0

    for layer in range(num_layers):
        # Data re-uploading
        for i, q in enumerate(qubits):
            circuit.append(cirq.rx(data_symbols[i])(q))

        # Variational layer
        for i, q in enumerate(qubits):
            circuit.append(cirq.ry(theta_symbols[theta_idx])(q))
            theta_idx += 1

        # Entanglement
        for i in range(num_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    # Correlator readout
    readout_ops = []
    for i in range(num_qubits - 1):
        readout_ops.append(cirq.Z(qubits[i]) * cirq.Z(qubits[i + 1]))
    readout_ops.append(cirq.Z(qubits[-1]))

    return circuit, qubits, data_symbols, theta_symbols, readout_ops



# ============================================================
# Quantum latent layer (TFQ)
# ============================================================
class QuantumLatentLayer(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        (
            self.circuit,
            self.qubits,
            self.x_symbols,
            self.theta_symbols,
            self.readout_ops,
        #) = build_latent_circuit(latent_dim)
        ) = create_reuploading_pqc_with_multiqubit_correlators(
            num_qubits=latent_dim,
            num_layers=3,
        )

        # Trainable quantum parameters
        '''
        self.theta = tf.Variable(
            initial_value=tf.random.normal([latent_dim]),
            trainable=True,
            dtype=tf.float32,
            name="quantum_theta",
        )
        '''
        self.theta = tf.Variable(
            initial_value=tf.random.normal([len(self.theta_symbols)]),
            trainable=True,
            dtype=tf.float32,
        )

        self.readout_ops = [cirq.Z(q) for q in self.qubits]
        self.expectation = tfq.layers.Expectation()

        '''
        self.symbol_names = (
            [str(s) for s in self.x_symbols]
            + [str(s) for s in self.theta_symbols]
        )
        '''
        self.symbol_names = (
            [str(s) for s in self.x_symbols]
            + [str(s) for s in self.theta_symbols]
        )


    def call(self, x):
        """
        x shape: (batch_size, latent_dim)
        returns: (batch_size, latent_dim)
        """
        batch_size = tf.shape(x)[0]

        circuits = tf.repeat(
            tfq.convert_to_tensor([self.circuit]),
            repeats=batch_size,
        )

        theta_batch = tf.repeat(
            self.theta[None, :], repeats=batch_size, axis=0
        )

        symbol_values = tf.concat([x, theta_batch], axis=1)

        return self.expectation(
            circuits,
            symbol_names=self.symbol_names,
            symbol_values=symbol_values,
            operators=self.readout_ops,
        )


# ============================================================
# QVAE model
# ============================================================

class TFQQVAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, model_name="TFQQVAE"):
        super().__init__(name=model_name)
        self.latent_dim = latent_dim

        # Encoder (classical)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(latent_dim * 2),
        ])

        # Quantum latent space
        self.quantum_latent = QuantumLatentLayer(latent_dim)

        # Decoder (classical)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(input_dim),
        ])

    def call(self, x):
        mean_log_var = self.encoder(x)
        mean, log_var = tf.split(mean_log_var, 2, axis=1)

        # Quantum latent representation
        z_q = self.quantum_latent(mean)

        # Decode
        x_hat = self.decoder(z_q)

        return x_hat, mean, log_var


# ============================================================
# Training loop
# ============================================================
def train_QVAE(
    model,
    dataset,
    epochs=50,
    learning_rate=1e-4,
    model_name="UNK",
    save_dir="output_qvae",
):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    history_loss = []
    history_reco = []
    history_qloss = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_reco = 0.0
        epoch_qloss = 0.0
        num_batches = 0

        for batch in dataset:
            with tf.GradientTape() as tape:
                x_hat, _, _ = model(batch)

                # reconstruction loss
                reco_loss = mse_loss_fn(batch, x_hat)

                # quantum regularization
                q_reg = 1e-4 * tf.reduce_sum(tf.square(model.quantum_latent.theta))

                # total loss
                loss = reco_loss + q_reg

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # accumulate per epoch
            epoch_loss += loss.numpy()
            epoch_reco += reco_loss.numpy()
            epoch_qloss += q_reg.numpy()
            num_batches += 1

        # average over batches
        epoch_loss /= num_batches
        epoch_reco /= num_batches
        epoch_qloss /= num_batches

        history_loss.append(epoch_loss)
        history_reco.append(epoch_reco)
        history_qloss.append(epoch_qloss)

        print(
            f"[QVAE {model_name}] Epoch {epoch+1}/{epochs} | "
            f"Reco: {epoch_reco:.6f}  | "
            f"Quantum Loss: {epoch_qloss:.6f} | "
            f"Total: {epoch_loss:.6f}"
        )

    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        f"{save_dir}/QVAE_{model_name}.npz",
        loss=np.array(history_loss),
        reco_loss=np.array(history_reco),
        qloss=np.array(history_qloss),
    )

    print(f"[QVAE {model_name}] Saved training history")


# ============================================================
# Convenience training wrapper
# ============================================================
def train_on_subset(
    X_subset,
    latent_dim=4,
    epochs=50,
    model_name="A",
    save_dir="output_qvae",
):
    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, f"qvae_{model_name}.h5")

    model = TFQQVAE(
        input_dim=X_subset.shape[1],
        latent_dim=latent_dim,
        model_name=model_name,
    )

    if os.path.exists(weight_path):
        dummy = tf.convert_to_tensor(X_subset[:1], dtype=tf.float32)
        model(dummy)
        model.load_weights(weight_path)
        print(f"Loaded existing model: {weight_path}")
        return model

    dataset = tf.data.Dataset.from_tensor_slices(
        X_subset.astype(np.float32)
    )
    dataset = dataset.shuffle(
        buffer_size=len(X_subset), seed=47
    ).batch(8)

    print(
        f"Training QVAE {model_name} on {len(X_subset)} samples "
        f"with {latent_dim} qubits"
    )

    train_QVAE(
        model,
        dataset,
        epochs=epochs,
        learning_rate=1e-4,
        model_name=model_name,
        save_dir=save_dir,
    )

    model.save_weights(weight_path)
    print(f"Model weights saved: {weight_path}")

    return model
