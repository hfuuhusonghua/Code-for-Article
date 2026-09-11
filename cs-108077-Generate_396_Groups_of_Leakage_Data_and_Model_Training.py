python
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------
# Step 1: Generate 396 groups of leakage data (consistent with prompt)
# --------------------------
# Leakage conditions: 66 pipes × 6 leakage flows (5,8,10,15,20,25 L/s) = 396 groups
leakage_flows = [5, 8, 10, 15, 20, 25]  # 6 leakage cases
n_pipes_xc = 66  # 66 pipes in Xuancheng water network (Fig.5)
data_list = []

# Get pipe length for each pipe (to calculate leakage distance: center of pipe)
pipe_lengths = {}
for pipe in pipe_data_xuancheng:
    pipe_id, from_node, to_node, diameter, length = pipe
    pipe_lengths[pipe_id] = length

# Simulate each leakage condition
for pipe_id in range(1, n_pipes_xc + 1):
    # Get pipe connection nodes to determine leakage node (center of pipe, simulate as a virtual node)
    pipe_info = [p for p in pipe_data_xuancheng if p[0] == pipe_id][0]
    from_node, to_node = pipe_info[1], pipe_info[2]
    leakage_distance = pipe_lengths[pipe_id] / 2  # Leakage at center of pipe

    for flow in leakage_flows:
        # Run PDD simulation for current leakage condition
        leak_pressure = run_xc_pdd_simulation(inp_path_xc, leakage_node=from_node, leakage_flow=flow)
        # Calculate pressure change rate (compared with normal condition)
        normal_pressure = run_xc_pdd_simulation(inp_path_xc, leakage_node=None, leakage_flow=0)
        pressure_change = (leak_pressure - normal_pressure) / normal_pressure  # Pressure change rate

        # Extract 10 time steps of pressure change rate (for STNN input)
        pressure_change_sample = pressure_change.iloc[:10].values  # 10 time steps, 5 monitoring points
        # Append to data list: [pressure_change, pipe_id, leakage_distance, leakage_flow]
        data_list.append([pressure_change_sample, pipe_id, leakage_distance, flow])

# Convert to numpy array
data_array = np.array(data_list, dtype=object)
X_xc = np.stack(data_array[:, 0])  # Input features: (396, 10, 5)
y_pipe_xc = np.array(data_array[:, 1], dtype=int)  # Leakage pipe ID
y_dist_xc = np.array(data_array[:, 2], dtype=float)  # Leakage distance
y_flow_xc = np.array(data_array[:, 3], dtype=int)  # Leakage flow

# Save raw data
np.savez("xuancheng_396_leak_data.npz", X=X_xc, y_pipe=y_pipe_xc, y_dist=y_dist_xc, y_flow=y_flow_xc)
print("\n396 groups of leakage data generated and saved (consistent with prompt)")

# --------------------------
# Step 2: Data Split (330 for training, 66 for verification with 15L/s leakage)
# --------------------------
# Filter data with leakage flow = 15L/s (for verification)
verify_mask = y_flow_xc == 15
X_verify = X_xc[verify_mask]
y_pipe_verify = y_pipe_xc[verify_mask]
y_dist_verify = y_dist_xc[verify_mask]

# Training data: remaining 330 groups (exclude 66 groups of 15L/s)
train_mask = ~verify_mask
X_train_xc = X_xc[train_mask]
y_pipe_train_xc = y_pipe_xc[train_mask]
y_dist_train_xc = y_dist_xc[train_mask]

# Data normalization
scaler_xc = StandardScaler()
X_train_xc = scaler_xc.fit_transform(X_train_xc.reshape(-1, X_train_xc.shape[-1])).reshape(X_train_xc.shape)
X_verify_xc = scaler_xc.transform(X_verify.reshape(-1, X_verify.shape[-1])).reshape(X_verify.shape)

print(f"Training data: {len(X_train_xc)} groups, Verification data: {len(X_verify_xc)} groups (15L/s leakage)")


# --------------------------
# Step 3: STNN Model Training (based on Table11 parameters)
# --------------------------
# Reuse STNN model structure (consistent with Table11)
def build_xc_stnn_model(input_shape, n_pipes):
    time_input = Input(shape=input_shape, name="time_input")
    time_conv1 = Conv1D(filters=32, kernel_size=3, activation='gelu', padding='same')(time_input)
    time_conv2 = Conv1D(filters=32, kernel_size=3, activation='gelu', padding='same')(time_conv1)
    time_pool = MaxPooling1D(pool_size=2)(time_conv2)

    time_conv3 = Conv1D(filters=64, kernel_size=3, activation='gelu', padding='same')(time_pool)
    time_conv4 = Conv1D(filters=64, kernel_size=3, activation='gelu', padding='same')(time_conv3)
    time_pool2 = MaxPooling1D(pool_size=2)(time_conv4)

    time_conv5 = Conv1D(filters=128, kernel_size=3, activation='gelu', padding='same')(time_pool2)
    time_conv6 = Conv1D(filters=128, kernel_size=3, activation='gelu', padding='same')(time_conv5)
    time_pool3 = MaxPooling1D(pool_size=2)(time_conv6)

    time_flatten = Flatten()(time_pool3)

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

    fused = Concatenate()([time_flatten, space_flatten])

    hidden1 = Dense(300, activation='relu')(fused)
    dropout1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(100, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(hidden2)
    hidden3 = Dense(50, activation='relu')(dropout2)
    dropout3 = Dropout(0.5)(hidden3)

    output_pipe = Dense(n_pipes, activation='softmax', name="pipe_output")(dropout3)
    output_dist = Dense(1, activation='linear', name="distance_output")(dropout3)

    model = Model(inputs=[time_input, space_input], outputs=[output_pipe, output_dist])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"pipe_output": "sparse_categorical_crossentropy", "distance_output": "mse"},
        metrics={"pipe_output": "accuracy", "distance_output": "mse"}
    )
    return model


# Initialize and train model
input_shape_xc = (X_train_xc.shape[1], X_train_xc.shape[2])  # (10, 5)
xc_stnn_model = build_xc_stnn_model(input_shape_xc, n_pipes_xc)

history_xc = xc_stnn_model.fit(
    [X_train_xc, X_train_xc],
    [y_pipe_train_xc - 1, y_dist_train_xc],  # Pipe ID 0-based
    epochs=5000,  # Consistent with Table11
    batch_size=128,  # Consistent with Table3
    validation_data=([X_verify_xc, X_verify_xc], [y_pipe_verify - 1, y_dist_verify]),
    verbose=1
)

# Save model
xc_stnn_model.save("xuancheng_stnn_model.h5")
print("\nXuancheng STNN model saved successfully")

# --------------------------
# Step 4: Model Verification (15L/s leakage, 66 groups of data)
# --------------------------
# Predict on verification set
y_pipe_pred_prob, y_dist_pred = xc_stnn_model.predict([X_verify_xc, X_verify_xc], verbose=0)
y_pipe_pred = np.argmax(y_pipe_pred_prob, axis=1) + 1  # Convert to 1-based pipe ID

# Calculate verification metrics (consistent with Table11)
pipe_accuracy = np.mean(y_pipe_pred == y_pipe_verify)
dist_mse = np.mean((y_dist_pred - y_dist_verify) ** 2)
dist_mae = np.mean(np.abs(y_dist_pred - y_dist_verify))

# Output verification results
print("\nModel Verification Results (15L/s leakage, 66 groups):")
print(f"Leakage Pipe Localization Accuracy: {pipe_accuracy:.4f}")
print(f"Leakage Distance MSE: {dist_mse:.4f}")
print(f"Leakage Distance MAE: {dist_mae:.4f}")

# Save verification results (consistent with Table11 format)
verification_results = pd.DataFrame({
    "Pipe ID (True)": y_pipe_verify,
    "Pipe ID (Pred)": y_pipe_pred,
    "Distance (True, m)": y_dist_verify.round(3),
    "Distance (Pred, m)": y_dist_pred.flatten().round(3),
    "Distance Error (m)": np.abs(y_dist_pred.flatten() - y_dist_verify).round(3)
})
verification_results.to_csv("xuancheng_model_verification.csv", index=False, encoding="utf