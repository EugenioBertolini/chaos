import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
import pyEDM as edm


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


def plt_corrs(all_df, col, nlags, e, tau, r, ax, colors):
    acf_ = acf(all_df[col], nlags=nlags)
    pred_interval_df = edm.PredictInterval(
        dataFrame=all_df,
        columns=col,
        target=col,
        lib=[1, le // 2],  # pyright: ignore
        pred=[le // 2, le],  # pyright: ignore
        maxTp=nlags,
        E=e,
        tau=-tau,
        exclusionRadius=r,
        showPlot=False,
        validLib=[],
        numProcess=4,
    )
    ax.plot(np.abs(acf_), "o-", label=f"corr {col}", color=colors[0])
    ax.plot(
        pred_interval_df["Tp"],
        pred_interval_df["rho"],
        "--",
        label=f"pred interval {col}",
        color=colors[1],
    )


pos, vel, acc = load_kinematics_data()
pos = (pos - pos.mean(axis=0)) / pos.std(axis=0)
vel = (vel - vel.mean(axis=0)) / vel.std(axis=0)
acc = (acc - acc.mean(axis=0)) / acc.std(axis=0)

pos_df = pd.DataFrame(pos, columns=pd.Index([f"pos{ch}" for ch in range(pos.shape[1])]))
vel_df = pd.DataFrame(vel, columns=pd.Index([f"vel{ch}" for ch in range(vel.shape[1])]))
acc_df = pd.DataFrame(acc, columns=pd.Index([f"acc{ch}" for ch in range(acc.shape[1])]))

ch = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
st = 81000
le = 4000
step = 1
pos_df = pos_df.iloc[st : st + le : step]
vel_df = vel_df.iloc[st : st + le : step]
acc_df = acc_df.iloc[st : st + le : step]
all_df = pd.concat([pos_df, vel_df, acc_df], axis=1)

print(f"{pos_df.shape=}")
print(f"{vel_df.shape=}")
print(f"{acc_df.shape=}")
print(f"{all_df.shape=}")

nlags = 20
c = sns.color_palette("Paired")
fig, ax = plt.subplots(3, 2, figsize=(18, 9))
for ch_i, ax_i in zip(ch, ax.flatten()):
    print(f"\n{ch_i=}")
    plt_corrs(all_df, f"pos{ch_i[0]}", nlags, 6, 4, 25, ax_i, c[:2])
    plt_corrs(all_df, f"pos{ch_i[1]}", nlags, 6, 4, 25, ax_i, c[2:4])
    plt_corrs(all_df, f"vel{ch_i[0]}", nlags, 6, 4, 25, ax_i, c[4:6])
    plt_corrs(all_df, f"vel{ch_i[1]}", nlags, 6, 4, 25, ax_i, c[6:8])
    plt_corrs(all_df, f"acc{ch_i[0]}", nlags, 6, 4, 25, ax_i, c[8:10])
    plt_corrs(all_df, f"acc{ch_i[1]}", nlags, 6, 4, 25, ax_i, c[10:])

    ax_i.set_xlim(0, nlags + nlags // 3)
    ax_i.legend()

plt.show()
