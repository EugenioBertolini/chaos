import h5py
import numpy as np
import pandas as pd
import pyEDM as edm

import itertools
import multiprocessing
from joblib import Parallel, delayed


# --- Optimization Parameters ---
E_MAX = 12
TAU_RANGE = (2, 21)
TP_RANGE = (4, 17, 2)
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

    for tau in range(*TAU_RANGE):
        tau_idx = tau - TAU_RANGE[0]
        for tp in range(*TP_RANGE):
            tmp_df = edm.EmbedDimension(
                dataFrame=data_df,
                columns=col_name,
                target=col_name,
                lib=f"1 {LE // 2}",
                pred=f"{LE // 2 + 1} {LE}",
                maxE=E_MAX,
                Tp=tp,
                tau=-tau,
                exclusionRadius=25,
                showPlot=False,
                numProcess=1,
            )
            rho_matrix[tau_idx, :] += tmp_df["rho"].values  # pyright: ignore

    rho_matrix = rho_matrix / ((TP_RANGE[1] - TP_RANGE[0]) / TP_RANGE[2])
    tau_idx, e_idx = np.unravel_index(rho_matrix.argmax(), rho_matrix.shape)
    best_e = 1 + e_idx
    best_tau = TAU_RANGE[0] + tau_idx
    print(
        f"✅ Finished signal: {col_name}. Best E={best_e}, Best Tau={best_tau}",
        f"Rho={rho_matrix[tau_idx, e_idx]} == {rho_matrix.max()}",
    )
    return (best_e, best_tau)


def calculate_single_ccm(ch_in, ch_tar, summary_df, data_df, lib_sizes, sample_size):
    """
    Worker function to compute CCM for one pair of signals.

    Args:
        ch_in (str): The name of the input/library signal.
        ch_tar (str): The name of the target signal.
        summary_df (pd.DataFrame): DataFrame with optimal E and tau for each signal.
        data_df (pd.DataFrame): DataFrame containing all time series.
        lib_sizes (list): List of library sizes to use for CCM.
        sample_size (int): Number of random libraries to create.

    Returns:
        tuple: A tuple containing (column_name, ccm_values_series).
    """
    print(f"Starting processing for signals: {ch_in}-->{ch_tar}...")
    params = summary_df.loc[ch_in]
    optimal_e = int(params["Optimal_E"])
    optimal_tau = int(params["Optimal_Tau"])

    ccm_result = edm.CCM(
        dataFrame=data_df,
        columns=ch_in,
        target=ch_tar,
        E=optimal_e,
        Tp=0,
        tau=-optimal_tau,
        libSizes=lib_sizes,
        sample=sample_size,
        seed=42,
        verbose=False,
        showPlot=False,
    )

    col_name = f"{ch_in}:{ch_tar}"
    ccm_values = ccm_result.iloc[:, 1]  # pyright: ignore

    return (col_name, ccm_values)


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

    # print(f"\nStarting parallel optimization for {len(cols)} signals...")
    # print(f"Using {multiprocessing.cpu_count()} CPU cores.")
    #
    # results = Parallel(n_jobs=-1, backend="loky")(
    #     delayed(find_best_embedding)(col, all_df) for col in cols
    # )
    # chosen_embeddings, chosen_taus = zip(*results)
    # chosen_embeddings = list(chosen_embeddings)
    # chosen_taus = list(chosen_taus)
    #
    # summary_df = pd.DataFrame(
    #     {"Signal": cols, "Optimal_E": chosen_embeddings, "Optimal_Tau": chosen_taus}
    # )
    # print("\n--- Optimization Complete ---")
    # summary_df.to_csv(
    #     f"bestE&tau-{LE}-Emax{E_MAX}-tau_min{TAU_RANGE[0]}-tau_max{TAU_RANGE[1]}.csv"
    # )
    # print(summary_df)

    summary_df = pd.read_csv(
        f"bestE&tau-{LE}-Emax{E_MAX}-tau_min{TAU_RANGE[0]}-tau_max{TAU_RANGE[1]}.csv",
        index_col=0,
    )
    if "Signal" in summary_df.columns:
        summary_df = summary_df.set_index("Signal")
    print("Loaded summary_df from file.")

    lib = [le // 48, le // 2 + le // 48, le // 24]
    sample = 5

    all_pairs = list(itertools.product(cols, repeat=2))
    print(f"\nPreparing to run CCM for {len(all_pairs)} signal pairs...")
    print(f"Using {multiprocessing.cpu_count()} CPU cores.")

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(calculate_single_ccm)(ch_in, ch_tar, summary_df, all_df, lib, sample)
        for ch_in, ch_tar in all_pairs
    )

    results_dict = {name: series for name, series in results}  # pyright: ignore
    ccm_values_df = pd.DataFrame(results_dict)

    lib_size_df = edm.CCM(
        dataFrame=all_df,
        columns=cols[0],  # pyright: ignore
        target=cols[1],  # pyright: ignore
        E=3,
        tau=2,
        libSizes=lib,  # pyright: ignore
        sample=1,
        showPlot=False,
    )[["LibSize"]]
    ccm_long = pd.concat([lib_size_df, ccm_values_df], axis=1)  # pyright: ignore

    ccm_long.to_csv(f"ccm-long-{LE}-optimal-E&tau-parallel-2.csv")
    print("\n--- All Done! ---")
