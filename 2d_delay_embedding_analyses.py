import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


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


def plt_colorlines(x, y, ax, xlabel, ylabel, title):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,  # pyright: ignore
        cmap="viridis",
        norm=plt.Normalize(0, len(x) - 1),  # pyright: ignore
    )
    lc.set_array(np.arange(len(x) - 1))
    lc.set_linewidth(1.5)
    ax.add_collection(lc)
    ax.scatter(
        x,
        y,
        c=np.arange(len(x)),
        cmap="viridis",
        s=10,
        edgecolors="black",
        linewidths=0.5,
        zorder=2,
    )
    fig.colorbar(lc, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


pos, vel, acc = load_kinematics_data()

pos = (pos - pos.mean(axis=0)) / pos.std(axis=0)
vel = (vel - vel.mean(axis=0)) / vel.std(axis=0)
acc = (acc - acc.mean(axis=0)) / acc.std(axis=0)

pos_df = pd.DataFrame(pos, columns=pd.Index([f"pos{ch}" for ch in range(pos.shape[1])]))
vel_df = pd.DataFrame(vel, columns=pd.Index([f"vel{ch}" for ch in range(vel.shape[1])]))
acc_df = pd.DataFrame(acc, columns=pd.Index([f"acc{ch}" for ch in range(acc.shape[1])]))

ch = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
st = 81400
le = 1000
step = 1
pos_df = pos_df.iloc[st : st + le : step]
vel_df = vel_df.iloc[st : st + le : step]
acc_df = acc_df.iloc[st : st + le : step]
all_df = pd.concat([pos_df, vel_df, acc_df], axis=1)

print(f"{pos_df.shape=}")
print(f"{vel_df.shape=}")
print(f"{acc_df.shape=}")
print(f"{all_df.shape=}")

lag = 4

fig, ax = plt.subplots(2, 4, figsize=(15, 3))

plt_colorlines(
    all_df[f"vel{ch[0][0]}"][lag:],
    all_df[f"vel{ch[0][1]}"][lag:],
    ax=ax[0, 0],
    xlabel="x leg0",
    ylabel="y leg0",
    title="x vs y leg0",
)
plt_colorlines(
    all_df[f"vel{ch[0][0]}"][lag:],
    all_df[f"vel{ch[0][1]}"][:-lag],
    ax=ax[0, 1],
    xlabel="x leg0",
    ylabel="lag y leg0",
    title="lag x vs y leg0",
)
plt_colorlines(
    all_df[f"vel{ch[0][0]}"][lag:],
    all_df[f"vel{ch[1][0]}"][lag:],
    ax=ax[0, 2],
    xlabel="x leg0",
    ylabel="x leg1",
    title="x leg0 vs leg1",
)
plt_colorlines(
    all_df[f"vel{ch[0][0]}"][lag:],
    all_df[f"vel{ch[1][0]}"][:-lag],
    ax=ax[0, 3],
    xlabel="x leg0",
    ylabel="lag x leg1",
    title="lag x leg0 vs leg1",
)

plt_colorlines(
    all_df[f"vel{ch[1][0]}"][lag:],
    all_df[f"vel{ch[1][1]}"][lag:],
    ax=ax[1, 0],
    xlabel="x leg1",
    ylabel="y leg1",
    title="x vs y leg1",
)
plt_colorlines(
    all_df[f"vel{ch[1][0]}"][lag:],
    all_df[f"vel{ch[1][1]}"][:-lag],
    ax=ax[1, 1],
    xlabel="x leg1",
    ylabel="lag y leg1",
    title="lag x vs y leg1",
)
plt_colorlines(
    all_df[f"vel{ch[0][1]}"][lag:],
    all_df[f"vel{ch[1][1]}"][lag:],
    ax=ax[1, 2],
    xlabel="y leg0",
    ylabel="y leg1",
    title="y leg0 vs leg1",
)
plt_colorlines(
    all_df[f"vel{ch[0][1]}"][lag:],
    all_df[f"vel{ch[1][1]}"][:-lag],
    ax=ax[1, 3],
    xlabel="y leg0",
    ylabel="lag y leg1",
    title="lag y leg0 vs leg1",
)

plt.show()
