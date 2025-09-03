import numpy as np
import pandas as pd
import os
import h5py
from numcodecs import Blosc
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def load_hdf5(filename=r".\keypoints_eks_all.h5"):
    print("\nLoading HDF5 file...")
    keypoints_dict = {
        3: {"abs": None, "center": None, "info": None, "rel": None, "stance": None},
        7: {"abs": None, "center": None, "info": None, "rel": None, "stance": None},
        11: {"abs": None, "center": None, "info": None, "rel": None, "stance": None},
        15: {"abs": None, "center": None, "info": None, "rel": None, "stance": None},
        24: {"abs": None, "center": None, "info": None, "rel": None, "stance": None},
    }
    listidxs = []
    with pd.HDFStore(filename) as store:
        for node in store.keys():
            parts = node.split("/")
            idx = int(parts[1].replace("session", ""))
            name = node[len("session011  ") :]
            keypoints_dict[idx][name] = store[node]

    return keypoints_dict


def main():
    """pyEDM examples."""

    kp_dict = load_hdf5()
    signal = kp_dict[11]["rel"]
    signal = (signal - signal.mean(axis=0)) / signal.std(axis=0)
    rel = signal.to_numpy()

    # rel_s = savgol_filter(rel, polyorder=2, window_length=3, deriv=0, axis=0)
    # plt.figure()
    # plt.plot(rel[:, 0], "o-", label="rel")
    # plt.plot(rel_s[:, 0], ".-", alpha=0.7, label="rel filtered")
    # plt.legend()
    # plt.show()
    # return

    rel_v = savgol_filter(rel, polyorder=2, window_length=3, deriv=1, axis=0)
    rel_a = savgol_filter(rel, polyorder=2, window_length=3, deriv=2, axis=0)

    print(
        f"rel.shape: {rel.shape}, rel_v.shape: {rel_v.shape}, rel_a.shape: {rel_a.shape}"
    )

    file_path = "kinematics11.h5"
    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("rel", data=rel, compression="gzip")
        hf.create_dataset("rel_v", data=rel_v, compression="gzip")
        hf.create_dataset("rel_a", data=rel_a, compression="gzip")

    print(f"\nArrays successfully saved to {os.path.abspath(file_path)}")


if __name__ == "__main__":
    main()
