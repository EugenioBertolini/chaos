import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def plot_rho_colormaps_by_col(emb_df, max_e, tau_range):
    """
    Generates 2x6 colormap grids for position, velocity, and acceleration data.
    This version avoids regular expressions.
    """
    for data_type in ["pos", "vel", "acc"]:
        fig, axes = plt.subplots(2, 6, figsize=(20, 7))
        for i in range(12):
            row, col = i % 2, i // 2
            ax = axes[row, col]

            # Hardcode the channel name prefix we're looking for
            channel_name = f"{data_type}{i}"
            col_prefix = f"rho_{channel_name}_tau"

            # Select all columns related to this specific channel
            channel_cols = [c for c in emb_df.columns if c.startswith(col_prefix)]
            if channel_cols == []:
                continue

            # Create a temporary DataFrame and melt it to a long format
            df_subset = emb_df[["E"] + channel_cols]
            df_long = pd.melt(
                df_subset, id_vars=["E"], var_name="params", value_name="rho"
            )

            # Extract tau by splitting the string, avoiding regex
            df_long["tau"] = df_long["params"].str.split("tau").str[1].astype(int)

            # Pivot the data to create a matrix for the heatmap
            pivot_df = df_long.pivot(index="E", columns="tau", values="rho")

            # Plot the heatmap
            im = ax.imshow(
                pivot_df,
                cmap="viridis",
                aspect="auto",
                origin="lower",
                vmin=0,
                vmax=1,
                extent=[tau_range.min() - 0.5, tau_range.max() + 0.5, 0.5, max_e + 0.5],
            )
            ax.set_xticks(tau_range)
            ax.set_yticks(np.arange(1, max_e + 1))
            ax.set_xlabel("Time Delay (tau)")
            ax.set_ylabel("Embedding Dimension (E)")
            ax.set_title(channel_name)
            fig.colorbar(im, ax=ax)

        plt.tight_layout()

    plt.show()


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

tp = 6
max_e = 20
tau_range = np.arange(-1, -41, -1)
emb_df = None
for col in all_df.columns:
    for tau in tau_range:
        print(f"{col=}, {tau=}")
        tmp_df = edm.EmbedDimension(
            dataFrame=all_df,
            columns=col,
            target=col,
            lib=[1, le // 2],  # pyright: ignore
            pred=[le // 2, le],  # pyright: ignore
            maxE=max_e,
            Tp=tp,
            tau=int(tau),
            exclusionRadius=25,
            showPlot=False,
            validLib=[],
            numProcess=4,
        )
        rho_col_name = f"rho_{col}_tau{tau}"
        tmp_df.rename(columns={"rho": rho_col_name}, inplace=True)
        if emb_df is None:
            emb_df = tmp_df
        else:
            emb_df = pd.merge(emb_df, tmp_df[["E", rho_col_name]], on="E", how="outer")

plot_rho_colormaps_by_col(emb_df, max_e, tau_range)
