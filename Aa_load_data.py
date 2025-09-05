import h5py
import numpy as np
import pandas as pd


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


def build_df(st, le):
    pos, vel, acc = load_kinematics_data()
    pos_df = pd.DataFrame(
        pos, columns=pd.Index([f"pos{ch}" for ch in range(pos.shape[1])])
    )
    vel_df = pd.DataFrame(
        vel, columns=pd.Index([f"vel{ch}" for ch in range(vel.shape[1])])
    )
    acc_df = pd.DataFrame(
        acc, columns=pd.Index([f"acc{ch}" for ch in range(acc.shape[1])])
    )

    step = 1
    pos_df = pos_df.iloc[st : st + le : step]
    vel_df = vel_df.iloc[st : st + le : step]
    acc_df = acc_df.iloc[st : st + le : step]
    all_df = pd.concat([pos_df, vel_df, acc_df], axis=1)
    all_df = (all_df - all_df.min(axis=0)) / (all_df.max(axis=0) - all_df.min(axis=0))
    all_df += 0.01

    print(f"{all_df.shape=}")

    return all_df


if __name__ == "__main__":
    st = 250000
    le = 24000

    new_le = 4000
    st = st + le // 2 - new_le // 2
    le = new_le

    build_df(st, le)
