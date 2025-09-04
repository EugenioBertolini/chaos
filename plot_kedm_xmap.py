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

emax = 36
tau = 4
tp = 6

xmap_matrix = np.load(f"kedm-xmap-{le}-Emax{emax}-tau{tau}-tp{tp}.npy")

corr_matrix = all_df.corr().abs()
diff = xmap_matrix - corr_matrix
diff[diff < 0] = 0

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
im0 = ax[0].imshow(
    all_df.corr(), cmap="RdBu", aspect="equal", origin="upper", vmin=-0.7, vmax=0.7
)
ax[0].set_title("Correlation Matrix")
fig.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(
    xmap_matrix, cmap="RdBu", aspect="equal", origin="upper", vmin=-1, vmax=1
)
ax[1].set_title("XMAP Matrix")
fig.colorbar(im1, ax=ax[1])
im2 = ax[2].imshow(diff, cmap="RdBu", aspect="equal", origin="upper", vmin=-1, vmax=1)
ax[2].set_title("Difference Matrix")
fig.colorbar(im2, ax=ax[2])
plt.tight_layout()
plt.show()
