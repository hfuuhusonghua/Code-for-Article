import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Step 1: Prepare Dataset (simulate pressure data for STNN training, consistent with paper Section 5.2)
# --------------------------
# Simulate dataset: 2000 samples per pipe (consistent with paper), 40 pipes, 3 leakage levels (large/medium/small)
np.random.seed(42)
n_pipes = 40
n_samples_per_pipe = 2000
n_sensors = len(sensor_nodes)  # 7 sensor nodes
leakage_levels = [5, 15, 25]  # Small:5L/s, Medium:15L/s, Large:25L/s (consistent with Table 6)

# Generate features (pressure change rate) and labels (leakage location: pipe ID + distance)
X = []
y_pipe = []  # Leakage pipe ID
y_distance = []  # Leakage distance (m)

for pipe_id in range(1, n_pipes + 1):
    for level in leakage_levels:
        for _ in range(n_samples_per_pipe // 3):
            # Simulate pressure change rate of 7 sensors (consistent with paper's 5 monitoring points, extended to 7)
            pressure_change = np.random.normal(loc=level * 0.01, scale=0.005,
                                               size=(10, n_sensors))  # 10 time steps, 7 sensors
            X.append(pressure_change)
            y_pipe.append(pipe_id)
            # Simulate leakage distance (random within pipe length, based on Table 1)
            pipe_length = [p[4] for p in pipe_data if p[0] == pipe_id][0]
            y_distance.append(np.random.uniform(0, pipe_length))

# Convert to numpy array
X = np.array(X)
y_pipe = np.array(y_pipe)
y_distance = np.array(y_distance)

# Data normalization (consistent with paper's data preprocessing)
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Split into training set and validation set (8:2, consistent with paper Section 4.3.2)
X_train, X_val, y_pipe_train, y_pipe_val, y_dist_train, y_dist_val = train_test_split(
    X, y_pipe, y_distance, test_size=0.2, random_state=42
)


# --------------------------
# Step 2: Build STNN Model (consistent with paper's structure in Section 4.1)
# --------------------------
def build_stnn_model(input_shape, n_pipes):
    # Time module: extract temporal features (bidirectional convolutional network)
    time_input = Input(shape=input_shape, name="time_input")
    time_conv1 = Conv1D(filters=32, kernel_size=3, activation='gelu', padding='same')(
        time_input)  # GELU activation (Section 4.2)
    time_conv2 = Conv1D(filters=32, kernel_size=3, activation='gelu', padding='same')(time_conv1)
    time_pool = MaxPooling1D(pool_size=2)(time_conv2)

    time_conv3 = Conv1D(filters=64, kernel_size=3, activation='gelu', padding='same')(time_pool)
    time_conv4 = Conv1D(filters=64, kernel_size=3, activation='gelu', padding='same')(time_conv3)
    time_pool2 = MaxPooling1D(pool_size=2)(time_conv4)

    time_conv5 = Conv1D(filters=128, kernel_size=3, activation='gelu', padding='same')(time_pool2)
    time_conv6 = Conv1D(filters=128, kernel_size=3, activation='gelu', padding='same')(time_conv5)
    time_pool3 = MaxPooling1D(pool_size=2)(time_conv6)

    time_flatten = Flatten()(time_pool3)

    # Space module: extract spatial features (multi-convolutional network)
    space_input = Input(shape=input_shape, name="space_input")
    space_conv1 = Conv1D(filters=32, kernel_size=3, activation='gelu', padding='same')(space_input)
    space_conv2 = Conv1D(filters=32, kernel_size=3, activation='gelu', padding='same')(space_conv1)
    space_pool = MaxPooling1D(pool_size=2)(space_conv2)

    space_conv3 = Conv1D(filters=64, kernel_size=3, activation='gelu', padding='same')(space_pool)
    space_conv4 = Conv1D(filters=64, kernel_size=3, activation='gelu', padding='same')(space_conv3)
    space_pool2 = MaxPooling1D(pool_size=2)(space_conv4)

    space_conv5 = Conv1D(filters=128, kernel_size=3, activation='gelu', padding='same')(space_pool2)
    space_conv6 = Conv1D(filters=128, kernel_size=3, activation='gelu', padding='same')(space_conv5)
    space_pool3 = MaxPooling1D(pool_size=2)(space_conv6)

    space_flatten = Flatten()(space_pool3)

    # Feature fusion
    fused = Concatenate()([time_flatten, space_flatten])

    # Hidden layers (3 layers, consistent with Table 11: 300, 100, 50 neurons)
    hidden1 = Dense(300, activation='relu')(fused)
    dropout1 = Dropout(0.5)(hidden1)  # Dropout 0.5 (Table 11)
    hidden2 = Dense(100, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(hidden2)
    hidden3 = Dense(50, activation='relu')(dropout2)
    dropout3 = Dropout(0.5)(hidden3)

    # Output layers: 1. Pipe ID classification; 2. Leakage distance regression
    output_pipe = Dense(n_pipes, activation='softmax', name="pipe_output")(dropout3)
    output_dist = Dense(1, activation='linear', name="distance_output")(dropout3)

    # Build model
    model = Model(inputs=[time_input, space_input], outputs=[output_pipe, output_dist])
    # Compile model (Adam optimizer, consistent with Table 11)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Learning rate 0.001 (Table 11)
        loss={
            "pipe_output": "sparse_categorical_crossentropy",
            "distance_output": "mse"
        },
        metrics={
            "pipe_output": "accuracy",
            "distance_output": "mse"
        }
    )
    return model


# Initialize model
input_shape = (X_train.shape[1], X_train.shape[2])  # (10 time steps, 7 sensors)
stnn_model = build_stnn_model(input_shape, n_pipes)
stnn_model.summary()

# --------------------------
# Step 3: Train STNN Model (consistent with Table 3 and Table 11)
# --------------------------
history = stnn_model.fit(
    [X_train, X_train],  # Time and space input use the same pressure data
    [y_pipe_train - 1, y_dist_train],  # Pipe ID is 0-based
    epochs=5000,  # Epochs=5000 (Table 11)
    batch_size=128,  # Batch size=128 (Table 3)
    validation_data=([X_val, X_val], [y_pipe_val - 1, y_dist_val]),
    verbose=1
)

# Save model
stnn_model.save("stnn_leakage_model.h5")
print("STNN model saved as 'stnn_leakage_model.h5'")

# --------------------------
# Step 4: Visualize Training Results (consistent with Fig.6 and Fig.7)
# --------------------------
# Fig.6: Model Training Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['pipe_output_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_pipe_output_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('STNN Model Training Accuracy (Consistent with Fig.6)')
plt.legend()
plt.grid(True)
plt.savefig("model_accuracy_fig.png", dpi=300)
plt.close()

# Fig.7: Model Training Loss Rate
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Total Training Loss')
plt.plot(history.history['val_loss'], label='Total Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('STNN Model Training Loss Rate (Consistent with Fig.7)')
plt.legend()
plt.grid(True)
plt.savefig("model_loss_fig.png", dpi=300)
plt.close()

print("Training accuracy and loss figures saved (consistent with Fig.6 and Fig.7)")