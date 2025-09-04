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

fig, ax = plt.subplots(3, 2, figsize=(15, 9))
for ch_i, ax_i in zip(ch, ax.flatten()):
    ax_i.plot(all_df[f"pos{ch_i[0]}"] + 5, label=f"ch{ch_i[0]}")
    ax_i.plot(all_df[f"pos{ch_i[1]}"] + 5, alpha=0.7, label=f"ch{ch_i[1]}")
    ax_i.plot(all_df[f"vel{ch_i[0]}"], label=f"vel ch{ch_i[0]}")
    ax_i.plot(all_df[f"vel{ch_i[1]}"], alpha=0.7, label=f"vel ch{ch_i[1]}")
    ax_i.plot(all_df[f"acc{ch_i[0]}"] - 5, label=f"acc ch{ch_i[0]}")
    ax_i.plot(all_df[f"acc{ch_i[1]}"] - 5, alpha=0.7, label=f"acc ch{ch_i[1]}")
    ax_i.set_ylim(-15, 10)
    ax_i.set_xlim(st, st + le + le // 3)
    ax_i.legend()
plt.show()
