import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

ccm_long = pd.read_csv(f"kedm-ccm-long-{le}-E{e}-tau{tau}-tp{tp}.csv", index_col=0)
print(ccm_long)
cols = ccm_long.columns

ndim = int(np.sqrt(ccm_long.shape[1]))
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
ax[0].set_title("Correlation Matrix")
fig.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(
    cm_matrix, cmap="RdBu", aspect="equal", origin="upper", vmin=-0.7, vmax=0.7
)
ax[1].set_title("CCM Matrix")
fig.colorbar(im1, ax=ax[1])
im2 = ax[2].imshow(
    diff, cmap="RdBu", aspect="equal", origin="upper", vmin=-0.7, vmax=0.7
)
ax[2].set_title("Difference Matrix")
fig.colorbar(im2, ax=ax[2])
plt.tight_layout()
plt.show()
