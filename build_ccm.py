import h5py
import numpy as np
import pandas as pd
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
tp = 6
tau = 4
lib = [le // 48, le // 2 + le // 48, le // 24]
sample = 5

cols = all_df.columns
cm_long = None
for i, ch_in in enumerate(cols):
    for j, ch_tar in enumerate(cols[i:]):
        cm = edm.CCM(
            dataFrame=all_df,
            columns=ch_in,
            target=ch_tar,
            E=e,
            Tp=tp,
            knn=0,
            tau=-tau,
            exclusionRadius=25,
            libSizes=lib,  # pyright: ignore
            sample=sample,
            seed=42,
            verbose=False,
            showPlot=False,
        )

        print(f"{ch_in} -> {ch_tar}")
        print(cm, "\n")

        if cm_long is None:
            cm_long = cm
        else:
            cm_long = pd.concat([cm_long, cm.iloc[:, 1:]], axis=1)  # pyright: ignore

cm_long.to_csv(f"ccm-long-{le}-E{e}-tau{tau}-tp{tp}.csv")  # pyright: ignore
