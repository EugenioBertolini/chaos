import h5py
import numpy as np
import pandas as pd
import kedm


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

e_max = all_df.shape[1]
e = 6
tp = 6
tau = 4
lib = np.arange(le // 48, le // 2 + le // 48, le // 24)
sample = 5

cols = all_df.columns
embed_dims = np.zeros((36, 1))
for i, ch in enumerate(cols):
    timeseries = all_df[ch]
    embedding_dimensions = kedm.edim(  # pyright: ignore
        all_df[ch],
        E_max=e_max,
        tau=tau,
        Tp=tp,
    )
    print(f"{ch = } --> E = {embedding_dimensions}")
    embed_dims[i] = embedding_dimensions

embed_dim_df = pd.DataFrame(embed_dims.T, columns=cols, index=pd.Index(["E"]))
embed_dim_df.to_csv(f"kedm-bestE-{le}-Emax{e_max}-tau{tau}-tp{tp}.csv")

ccm_all = np.zeros((len(lib), len(cols) ** 2))
for i, ch in enumerate(cols):
    for j, ch2 in enumerate(cols):
        ccm_two = kedm.ccm(  # pyright: ignore
            lib=all_df[ch],
            target=all_df[ch2],
            lib_sizes=lib,
            sample=sample,
            E=e,
            tau=tau,
            Tp=tp,
            seed=42,
            accuracy=0.999,
        )
        print(f"{ch} x {ch2} --> ccm = {np.round(ccm_two, 2)}")
        ccm_all[:, i * len(cols) + j] = np.asarray(ccm_two)

ccm_df = pd.DataFrame(
    ccm_all, columns=pd.Index([f"{ch}:{ch2}" for ch in cols for ch2 in cols]), index=lib
)
ccm_df.to_csv(f"kedm-ccm-long-{le}-E{e}-tau{tau}-tp{tp}.csv")

xmap_all = kedm.xmap(  # pyright: ignore
    all_df,
    embed_dims,
    tau=tau,
    Tp=tp,
)
print(xmap_all)

np.save(f"kedm-xmap-{le}-Emax{e_max}-tau{tau}-tp{tp}.npy", xmap_all)
