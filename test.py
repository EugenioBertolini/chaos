import h5py
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import pyEDM as edm
import pandas as pd


def load_kinematics_data(file_path="kinematics11.h5"):
    """
    Loads velocity and acceleration data from an HDF5 file.
    """
    print(f"Loading data from {file_path}...")
    with h5py.File(file_path, "r") as hf:
        vel = hf["rel_v"][:]
        acc = hf["rel_a"][:]
    print("Data loaded successfully.")
    return vel, acc


def plot_rho_colormaps(emb_df: pd.DataFrame):
    if emb_df is None or emb_df.empty:
        print("Input DataFrame is empty. Cannot generate plots.")
        return

    print("\nGenerating colormaps for forecast skill (rho)...")

    # Reshape the data from a wide to a long format for easier plotting
    id_vars = ["E"]
    value_vars = [col for col in emb_df.columns if col.startswith("rho_")]

    if not value_vars:
        print("No 'rho' columns found in the DataFrame to plot.")
        return

    long_df = pd.melt(
        emb_df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="params",
        value_name="rho",
    )

    # Extract tau and tp from the 'params' column (e.g., from 'rho_tau1_tp2')
    try:
        long_df["tau"] = [
            int(s[7 : s.find("tp") - 1]) for s in long_df["params"].to_list()
        ]
        long_df["tp"] = [
            int(s[s.find("tp") + 2 :]) for s in long_df["params"].to_list()
        ]
    except Exception as e:
        print(f"Could not parse 'tau' and 'tp' from column names. Error: {e}")
        return

    # Create a separate plot for each unique Tp value
    unique_tp = sorted(long_df["tp"].unique())

    fig, axes = plt.subplots(
        1, len(unique_tp), figsize=(6 * len(unique_tp), 5), squeeze=False
    )

    for i, tp in enumerate(unique_tp):
        ax = axes[0, i]
        df_for_plot = long_df[long_df["tp"] == tp]

        # Pivot the data to create a matrix for the heatmap
        pivot_df = df_for_plot.pivot(index="E", columns="tau", values="rho")

        im = ax.imshow(
            pivot_df,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            extent=[
                pivot_df.columns.min() - 0.5,
                pivot_df.columns.max() + 0.5,
                pivot_df.index.min() - 0.5,
                pivot_df.index.max() + 0.5,
            ],
        )

        fig.colorbar(im, ax=ax, label="Forecast Skill (rho)")
        ax.set_title(f"Forecast Skill (rho) for Tp = {tp}")
        ax.set_xlabel("Time Delay (tau)")
        ax.set_ylabel("Embedding Dimension (E)")

        ax.set_xticks(pivot_df.columns)
        ax.set_yticks(pivot_df.index)

    plt.tight_layout()
    plt.show()
    print("Colormaps displayed.")


vel, acc = load_kinematics_data()
vel = (vel - vel.mean(axis=0)) / vel.std(axis=0)
acc = (acc - acc.mean(axis=0)) / acc.std(axis=0)

vel_df = pd.DataFrame(vel, columns=[f"ch{ch}" for ch in range(vel.shape[1])])
acc_df = pd.DataFrame(acc, columns=[f"ch{ch}" for ch in range(acc.shape[1])])

ch = 0
st = 81400
le = 375 * 2
step = 1
vel_df = vel_df.iloc[st : st + le : step]
acc_df = acc_df.iloc[st : st + le : step]

print(f"{vel_df.shape=}")
print(f"{acc_df.shape=}")

lag = 4

fig, ax = plt.subplots(2, 2, figsize=(15, 3))
ax[0, 0].plot(vel_df[f"ch{ch}"][lag:], vel_df[f"ch{ch+1}"][lag:], "o-", alpha=0.7)
# ax[0].plot(vel_df[f"ch{ch}"], label=f"ch{ch}")
# ax[0].plot(vel_df[f"ch{ch+1}"], alpha=0.7, label=f"ch{ch+1}")
# ax[0].set_ylim(-15, 15)
# ax[0].legend()
ax[0, 0].set_xlabel("x_leg0")
ax[0, 0].set_ylabel("y_leg0")
ax[0, 0].set_aspect("equal")
ax[0, 0].set_title("velocity leg0")

ax[0, 1].plot(vel_df[f"ch{ch}"][:-lag], vel_df[f"ch{ch+1}"][lag:], "o-", alpha=0.7)
ax[0, 1].set_xlabel("x_leg0")
ax[0, 1].set_ylabel("y_leg0")
ax[0, 1].set_aspect("equal")
ax[0, 1].set_title(f"lagged velocity leg0 [{lag=}]")

ax[1, 0].plot(vel_df[f"ch{ch+2}"][lag:], vel_df[f"ch{ch+3}"][lag:], "o-", alpha=0.7)
# ax[1].plot(vel_df[f"ch{ch+2}"], label=f"ch{ch+2}")
# ax[1].plot(vel_df[f"ch{ch+3}"], alpha=0.7, label=f"ch{ch+3}")
# ax[1].set_ylim(-15, 15)
# ax[1].legend()
ax[1, 0].set_xlabel("x_leg1")
ax[1, 0].set_ylabel("y_leg1")
ax[1, 0].set_aspect("equal")
ax[1, 0].set_title("velocity leg1")

ax[1, 1].plot(vel_df[f"ch{ch+2}"][:-lag], vel_df[f"ch{ch+3}"][lag:], "o-", alpha=0.7)
ax[1, 1].set_xlabel("x_leg1")
ax[1, 1].set_ylabel("y_leg1")
ax[1, 1].set_aspect("equal")
ax[1, 1].set_title(f"lagged velocity leg1 [{lag=}]")

plt.show()
