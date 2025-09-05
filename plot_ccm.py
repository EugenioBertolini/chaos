import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


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

ccm_long = pd.read_csv("ccm-long-24000-optimal-E&tau-parallel-2.csv", index_col=0)
cols = ccm_long.columns[1:]

ndim = int(np.sqrt(len(cols)))
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
print(best)
cm_matrix = cm_matrix[best]

corr_matrix = all_df.corr().abs()
diff = cm_matrix - corr_matrix
diff[diff < 0] = 0

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
im0 = ax[0].imshow(
    all_df.corr(), cmap="RdBu", aspect="equal", origin="upper", vmin=-0.7, vmax=0.7
)
ax[0].set_xticks(np.arange(diff.shape[1]))
ax[0].set_yticks(np.arange(diff.shape[0]))
ax[0].set_xticklabels(all_df.columns, fontsize=8)
ax[0].set_yticklabels(all_df.columns, fontsize=8)
plt.setp(ax[0].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
ax[0].set_title("Correlation Matrix")

im1 = ax[1].imshow(
    cm_matrix, cmap="RdBu", aspect="equal", origin="upper", vmin=-0.7, vmax=0.7
)
ax[1].set_xticks(np.arange(diff.shape[1]))
ax[1].set_yticks(np.arange(diff.shape[0]))
ax[1].set_xticklabels(all_df.columns, fontsize=8)
ax[1].set_yticklabels(all_df.columns, fontsize=8)
plt.setp(ax[1].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
ax[1].set_title("CCM Matrix")

im2 = ax[2].imshow(
    diff, cmap="RdBu", aspect="equal", origin="upper", vmin=-0.7, vmax=0.7
)
ax[2].set_xticks(np.arange(diff.shape[1]))
ax[2].set_yticks(np.arange(diff.shape[0]))
ax[2].set_xticklabels(all_df.columns, fontsize=8)
ax[2].set_yticklabels(all_df.columns, fontsize=8)
plt.setp(ax[2].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
ax[2].set_title("Difference Matrix")
fig.colorbar(im2, ax=ax[2])

plt.tight_layout()
plt.show()


cm_matrix_df = pd.DataFrame(cm_matrix, index=all_df.columns, columns=all_df.columns)
diff_matrix = cm_matrix_df - corr_matrix
diff_matrix[diff_matrix < 0] = 0

print("\n--- Step 1: Clustering on Causality (diff matrix) ---")
target_signal = "vel0"
target_idx = all_df.columns.get_loc(target_signal)
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
dendro_data = sch.dendrogram(diff_linkage, labels=all_df.columns, leaf_rotation=90)
plt.axhline(y=0.8, color="r", linestyle="--", label="Cluster Threshold")
plt.legend()
plt.tight_layout()
plt.show()

# --- *** NEW SECTION: Plot the reordered diff matrix *** ---
print("\nPlotting the reordered causality (diff) matrix based on clustering...")
# Get the order of leaves from the dendrogram calculation
leaves_order = dendro_data["leaves"]
# Reorder the diff matrix according to the dendrogram leaves
diff_matrix_reordered = diff_matrix.iloc[leaves_order, leaves_order]

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(diff_matrix_reordered, cmap="viridis", interpolation="nearest")
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Causality Strength (CCM - Correlation)", rotation=-90, va="bottom")
ax.set_xticks(np.arange(diff_matrix_reordered.shape[1]))
ax.set_yticks(np.arange(diff_matrix_reordered.shape[0]))
ax.set_xticklabels(diff_matrix_reordered.columns, fontsize=8)
ax.set_yticklabels(diff_matrix_reordered.index, fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
ax.set_title("Causality (diff) Matrix Reordered by Hierarchical Clustering")
fig.tight_layout()
plt.show()
