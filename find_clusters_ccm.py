import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import pyEDM


def load_kinematics_data(file_path="kinematics11.h5"):
    """
    Loads velocity and acceleration data from an HDF5 file.
    """
    print(f"Loading data from {file_path}...")
    with h5py.File(file_path, "r") as hf:
        pos = hf["rel"][:]  # pyright: ignore
        vel = hf["rel_v"][:]  # pyright: ignore
        acc = hf["rel_a"][:]  # pyright: ignore
    print("Data loaded successfully.")
    return np.asarray(pos), np.asarray(vel), np.asarray(acc)


pos, vel, acc = load_kinematics_data()
pos = (pos - pos.mean(axis=0)) / pos.std(axis=0)
vel = (vel - vel.mean(axis=0)) / vel.std(axis=0)
acc = (acc - acc.mean(axis=0)) / acc.std(axis=0)

pos_df = pd.DataFrame(pos, columns=pd.Index([f"pos{ch}" for ch in range(pos.shape[1])]))
vel_df = pd.DataFrame(vel, columns=pd.Index([f"vel{ch}" for ch in range(vel.shape[1])]))
acc_df = pd.DataFrame(acc, columns=pd.Index([f"acc{ch}" for ch in range(acc.shape[1])]))

ch = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
st = 250000
le = 24000
step = 1
pos_df = pos_df.iloc[st : st + le : step]
vel_df = vel_df.iloc[st : st + le : step]
acc_df = acc_df.iloc[st : st + le : step]
all_df = pd.concat([pos_df, vel_df, acc_df], axis=1)

print(f"{pos_df.shape=}")
print(f"{vel_df.shape=}")
print(f"{acc_df.shape=}")
print(f"{all_df.shape=}")

e = 6
tau = 4
tp = 6

ccm_long = pd.read_csv(f"ccm-long-{le}-E{e}-tau{tau}-tp{tp}.csv", index_col=0)
cols = ccm_long.columns[1:]

ndim = int(np.sqrt(ccm_long.shape[1] - 1))
cm_matrix = np.zeros((len(ccm_long), ndim, ndim))
print(cm_matrix.shape)
for i, c in enumerate(cols):
    j_str, k_str = c.split(":")
    j = int(j_str[3:])
    k = int(k_str[3:])
    jstr = j_str[:3]
    kstr = k_str[:3]
    if jstr == "vel":
        j += ndim // 3
    elif jstr == "acc":
        j += ndim // 3 * 2
    if kstr == "vel":
        k += ndim // 3
    elif kstr == "acc":
        k += ndim // 3 * 2
    cm_matrix[:, j, k] = ccm_long[c].values  # pyright: ignore

best = np.argmax(cm_matrix.mean(axis=(1, 2)))
print(f"Best library index: {best}")
cm_matrix = cm_matrix[best]

corr_matrix = all_df.corr().abs()
# Create a DataFrame for the cm_matrix for easier indexing
cm_matrix_df = pd.DataFrame(cm_matrix, index=all_df.columns, columns=all_df.columns)

# Calculate diff matrix, setting negative values (where corr > ccm) to 0
diff_matrix = cm_matrix_df - corr_matrix
diff_matrix[diff_matrix < 0] = 0

# --- 1. Hierarchical Clustering on Causality (`diff` matrix) ---
print("\n--- Step 1: Clustering on Causality (diff matrix) ---")
target_signal = "vel0"
target_idx = all_df.columns.get_loc(target_signal)

# Convert similarity (diff_matrix) to distance for clustering
# We use 1 - similarity. The values are already scaled between 0 and 1.
diff_distance_matrix = 1 - diff_matrix.values

# Compute the linkage matrix using the 'ward' method
diff_linkage = sch.linkage(
    diff_distance_matrix[np.triu_indices(ndim, k=1)], method="ward"
)

# Plot the dendrogram for visualization
plt.figure(figsize=(15, 8))
plt.title("Hierarchical Clustering Dendrogram of Causality (diff matrix)")
plt.xlabel("Signal Index")
plt.ylabel("Distance (1 - Causality)")
sch.dendrogram(diff_linkage, labels=all_df.columns, leaf_rotation=90)
plt.axhline(
    y=0.8, color="r", linestyle="--", label="Cluster Threshold"
)  # Example threshold
plt.legend()
plt.tight_layout()
plt.show()

# Flatten the clusters based on a distance threshold
# This threshold determines how "close" signals must be to be in the same cluster.
# You may need to adjust this value based on the dendrogram.
distance_threshold_causality = 0.8
causality_clusters = sch.fcluster(
    diff_linkage, distance_threshold_causality, criterion="distance"
)

# Find the cluster that our target signal belongs to
target_cluster_id = causality_clusters[target_idx]
print(f"'{target_signal}' belongs to causality cluster: {target_cluster_id}")

# Find all signals in the same cluster as the target
causally_linked_indices = np.where(causality_clusters == target_cluster_id)[0]
causally_linked_signals = all_df.columns[causally_linked_indices].tolist()

# Remove the target signal itself from the list
if target_signal in causally_linked_signals:
    causally_linked_signals.remove(target_signal)

print(f"Signals causally linked with '{target_signal}': {causally_linked_signals}")

# --- 2. Verification with Correlation Clustering ---
print("\n--- Step 2: Verifying Low Cross-Correlation Among Causal Signals ---")
if not causally_linked_signals:
    print("No other signals found in the same causality cluster. Exiting.")
else:
    # Convert correlation to distance
    corr_distance_matrix = 1 - corr_matrix.values
    corr_linkage = sch.linkage(
        corr_distance_matrix[np.triu_indices(ndim, k=1)], method="ward"
    )

    # Define a stricter threshold for correlation, as we want to separate even moderately correlated signals.
    distance_threshold_corr = 0.5
    correlation_clusters = sch.fcluster(
        corr_linkage, distance_threshold_corr, criterion="distance"
    )

    # Get the correlation cluster IDs for each of our causally linked signals
    causal_signal_indices = [all_df.columns.get_loc(s) for s in causally_linked_signals]
    causal_corr_cluster_ids = correlation_clusters[causal_signal_indices]

    # Check if these signals are in different correlation clusters
    final_signals_for_smap = []
    seen_corr_clusters = set()

    print("Checking for cross-correlation:")
    for signal, cluster_id in zip(causally_linked_signals, causal_corr_cluster_ids):
        if cluster_id not in seen_corr_clusters:
            final_signals_for_smap.append(signal)
            seen_corr_clusters.add(cluster_id)
            print(f"  - Keeping '{signal}' (in correlation cluster {cluster_id})")
        else:
            print(
                f"  - Discarding '{signal}' (already have a signal from correlation cluster {cluster_id})"
            )

    if len(final_signals_for_smap) == len(causally_linked_signals):
        print(
            "\n✅ Verification successful: All causally linked signals are in different correlation clusters."
        )
    else:
        print(
            f"\n⚠️ Verification result: Some signals were discarded due to high cross-correlation."
        )

    print(
        f"Final signals for S-Map analysis with '{target_signal}': {final_signals_for_smap}"
    )

# --- 3. S-Map Analysis ---
print("\n--- Step 3: S-Map Analysis ---")
if not final_signals_for_smap:
    print("No signals remaining for S-Map analysis.")
else:
    # Set library and prediction sections for S-map (e.g., first half and second half)
    lib_start = 1
    lib_end = le // 2
    pred_start = lib_end + 1
    pred_end = le

    # Run S-Map for each identified signal against the target
    for predictor_signal in final_signals_for_smap:
        print(
            f"\nRunning S-Map: '{predictor_signal}' (predictor) xmap '{target_signal}' (target)..."
        )

        # SMap returns a dataframe with results
        smap_output = pyEDM.SMap(
            dataFrame=all_df,
            lib=f"{lib_start} {lib_end}",
            pred=f"{pred_start} {pred_end}",
            E=e,
            Tp=tp,
            tau=tau,
            columns=f"{target_signal} {predictor_signal}",
            target=target_signal,
            verbose=False,
        )

        # The S-Map output contains time-varying Jacobian coefficients.
        # The column name 'd(target)/d(predictor)' shows the influence of the predictor on the target.
        jacobian_col = f"d({target_signal})/d({predictor_signal})"

        if jacobian_col in smap_output.columns:
            plt.figure(figsize=(12, 5))
            plt.plot(smap_output["time"], smap_output[jacobian_col])
            plt.title(
                f"S-Map Interaction Strength: {predictor_signal} → {target_signal}"
            )
            plt.xlabel("Time Step")
            plt.ylabel(f"Jacobian: d({target_signal})/d({predictor_signal})")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.show()
        else:
            print(f"Could not find Jacobian column '{jacobian_col}' in S-Map output.")
