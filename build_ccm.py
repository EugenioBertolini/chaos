import h5py
import numpy as np
import pandas as pd
import pyEDM as edm

import multiprocessing
from joblib import Parallel, delayed


# --- Optimization Parameters ---
E_MAX = 12
TAU_RANGE = (2, 20)
TP = 10
ST = 250000
LE = 24000


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


# 1. Define a "worker function"
# This function contains the logic to find the best E and tau for a SINGLE signal.
def find_best_embedding(col_name, data_df):
    """
    Performs the search for optimal E and tau for a given time series.

    Args:
        col_name (str): The name of the column (signal) to process.
        data_df (pd.DataFrame): The dataframe containing all signals.

    Returns:
        tuple: A tuple containing (best_E, best_tau).
    """
    print(f"Starting processing for signal: {col_name}...")

    rho_matrix = np.zeros((TAU_RANGE[1] - TAU_RANGE[0], E_MAX))

    # Loop over tau values
    for tau in range(*TAU_RANGE):
        tau_idx = tau - TAU_RANGE[0]
        tmp_df = edm.EmbedDimension(
            dataFrame=data_df,
            columns=col_name,
            target=col_name,
            lib=f"1 {LE // 2}",
            pred=f"{LE // 2 + 1} {LE}",
            maxE=E_MAX,
            Tp=TP,
            tau=-tau,
            exclusionRadius=25,
            showPlot=False,
            numProcess=1,  # Set to 1 to avoid nested parallelization
        )
        rho_matrix[tau_idx, :] = tmp_df["rho"].values  # pyright: ignore

    # Find the E and tau that give the maximum rho (forecast skill)
    e_idx, tau_idx = np.unravel_index(rho_matrix.T.argmax(), rho_matrix.shape)

    # Convert matrix indices back to E and tau values
    best_e = 1 + e_idx
    best_tau = TAU_RANGE[0] + tau_idx

    print(f"✅ Finished signal: {col_name}. Best E={best_e}, Best Tau={best_tau}")

    return (best_e, best_tau)


if __name__ == "__main__":
    pos, vel, acc = load_kinematics_data()
    pos = (pos - pos.mean(axis=0)) / pos.std(axis=0)
    vel = (vel - vel.mean(axis=0)) / vel.std(axis=0)
    acc = (acc - acc.mean(axis=0)) / acc.std(axis=0)

    pos_df = pd.DataFrame(
        pos, columns=pd.Index([f"pos{ch}" for ch in range(pos.shape[1])])
    )
    vel_df = pd.DataFrame(
        vel, columns=pd.Index([f"vel{ch}" for ch in range(vel.shape[1])])
    )
    acc_df = pd.DataFrame(
        acc, columns=pd.Index([f"acc{ch}" for ch in range(acc.shape[1])])
    )

    ch = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    st = ST
    le = LE
    step = 1
    pos_df = pos_df.iloc[st : st + le : step]
    vel_df = vel_df.iloc[st : st + le : step]
    acc_df = acc_df.iloc[st : st + le : step]
    all_df = pd.concat([pos_df, vel_df, acc_df], axis=1)

    print(f"{pos_df.shape=}")
    print(f"{vel_df.shape=}")
    print(f"{acc_df.shape=}")
    print(f"{all_df.shape=}")

    cols = all_df.columns

    print(f"\nStarting parallel optimization for {len(cols)} signals...")
    print(f"Using {multiprocessing.cpu_count()} CPU cores.")

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(find_best_embedding)(col, all_df) for col in cols
    )
    chosen_embeddings, chosen_taus = zip(*results)
    chosen_embeddings = list(chosen_embeddings)
    chosen_taus = list(chosen_taus)

    summary_df = pd.DataFrame(
        {"Signal": cols, "Optimal_E": chosen_embeddings, "Optimal_Tau": chosen_taus}
    )
    print("\n--- Optimization Complete ---")
    summary_df.to_csv(
        f"bestE&tau-{LE}-Emax{E_MAX}-tau_min{TAU_RANGE[0]}-tau_max{TAU_RANGE[1]}-tp{TP}.csv"
    )
    print(summary_df)


# cm_long = None
# for i, ch_in in enumerate(cols):
#     for j, ch_tar in enumerate(cols):
#         cm = edm.CCM(
#             dataFrame=all_df,
#             columns=ch_in,
#             target=ch_tar,
#             E=embed_dims[i, 0],
#             Tp=tp,
#             knn=0,
#             tau=-tau,
#             exclusionRadius=25,
#             libSizes=lib,  # pyright: ignore
#             sample=sample,
#             seed=42,
#             verbose=False,
#             showPlot=False,
#         )
#
#         print(f"{ch_in} -> {ch_tar}")
#         print(cm, "\n")
#
#         if cm_long is None:
#             cm_long = cm
#         else:
#             cm_long = pd.concat([cm_long, cm.iloc[:, 1:]], axis=1)  # pyright: ignore
#
# cm_long.to_csv(f"ccm-long-{le}-E{e}-tau{tau}-tp{tp}.csv")  # pyright: ignore
