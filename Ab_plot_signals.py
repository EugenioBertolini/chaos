import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pyEDM
import Aa_load_data as aa


def basic_plots(all_df, ch):
    _, ax = plt.subplots(3, 2, figsize=(15, 9))
    for ch_i, ax_i in zip(ch, ax.flatten()):
        ax_i.plot(all_df[f"pos{ch_i[0]}"] + 1, label=f"ch{ch_i[0]}")
        ax_i.plot(all_df[f"pos{ch_i[1]}"] + 1, alpha=0.7, label=f"ch{ch_i[1]}")
        ax_i.plot(all_df[f"vel{ch_i[0]}"], label=f"vel ch{ch_i[0]}")
        ax_i.plot(all_df[f"vel{ch_i[1]}"], alpha=0.7, label=f"vel ch{ch_i[1]}")
        ax_i.plot(all_df[f"acc{ch_i[0]}"] - 1, label=f"acc ch{ch_i[0]}")
        ax_i.plot(all_df[f"acc{ch_i[1]}"] - 1, alpha=0.7, label=f"acc ch{ch_i[1]}")
        ax_i.set_xlim(st, st + le)
        ax_i.set_ylim(-1, 2)
        ax_i.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def more_plots(all_df, ch):
    _, ax = plt.subplots(3, 2, figsize=(15, 9))
    for ch_i, ax_i in zip(ch, ax.T):
        ax_i[0].plot(all_df[f"pos{ch_i[0]}"], label=f"ch{ch_i[0]}")
        ax_i[0].plot(all_df[f"pos{ch_i[1]}"], alpha=0.7, label=f"ch{ch_i[1]}")
        ax_i[1].plot(all_df[f"vel{ch_i[0]}"], label=f"vel ch{ch_i[0]}")
        ax_i[1].plot(all_df[f"vel{ch_i[1]}"], alpha=0.7, label=f"vel ch{ch_i[1]}")
        ax_i[2].plot(all_df[f"acc{ch_i[0]}"], label=f"acc ch{ch_i[0]}")
        ax_i[2].plot(all_df[f"acc{ch_i[1]}"], alpha=0.7, label=f"acc ch{ch_i[1]}")
        ax_i[0].set_ylim(0, 1)
        ax_i[1].set_ylim(0, 1)
        ax_i[2].set_ylim(0, 1)
        ax_i[0].set_xlim(st, st + le)
        ax_i[1].set_xlim(st, st + le)
        ax_i[2].set_xlim(st, st + le)
        ax_i[0].legend(loc="upper right")
        ax_i[1].legend(loc="upper right")
        ax_i[2].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def simplex_plots(all_df, range_tau, is_save=False):
    for tau in range_tau:
        fig, ax = plt.subplots(3, 8, figsize=(32, 16))

        plot_delay_emb(all_df, "vel1", "vel1", tau, ax[0, 0])  # vel y front left
        plot_delay_emb(all_df, "vel1", "vel0", tau, ax[1, 0])
        plot_delay_emb(all_df, "vel1", "vel3", tau, ax[2, 0])

        plot_delay_emb(all_df, "pos1", "pos1", tau, ax[0, 1])  # pos y front left
        plot_delay_emb(all_df, "pos1", "pos0", tau, ax[1, 1])
        plot_delay_emb(all_df, "pos1", "pos3", tau, ax[2, 1])

        plot_delay_emb(all_df, "vel1", "pos1", tau, ax[0, 2])  # vel:pos front left
        plot_delay_emb(all_df, "vel1", "pos3", tau, ax[1, 2])
        plot_delay_emb(all_df, "vel0", "pos0", tau, ax[2, 2])

        plot_delay_emb(all_df, "vel1", "vel5", tau, ax[0, 3])  # vel front left:vel mid
        plot_delay_emb(all_df, "vel1", "vel7", tau, ax[1, 3])
        plot_delay_emb(all_df, "vel1", "vel6", tau, ax[2, 3])

        plot_delay_emb(all_df, "vel3", "vel3", tau, ax[0, 4])  # vel y front right
        plot_delay_emb(all_df, "vel3", "vel2", tau, ax[1, 4])
        plot_delay_emb(all_df, "vel3", "vel1", tau, ax[2, 4])

        plot_delay_emb(all_df, "pos3", "pos3", tau, ax[0, 5])  # pos y front right
        plot_delay_emb(all_df, "pos3", "pos2", tau, ax[1, 5])
        plot_delay_emb(all_df, "pos3", "pos1", tau, ax[2, 5])

        plot_delay_emb(all_df, "vel3", "pos3", tau, ax[0, 6])  # vel:pos front right
        plot_delay_emb(all_df, "vel3", "pos1", tau, ax[1, 6])
        plot_delay_emb(all_df, "vel2", "pos2", tau, ax[2, 6])

        plot_delay_emb(all_df, "vel3", "vel7", tau, ax[0, 7])  # vel front right:vel mid
        plot_delay_emb(all_df, "vel3", "vel5", tau, ax[1, 7])
        plot_delay_emb(all_df, "vel3", "vel4", tau, ax[2, 7])

        plt.tight_layout()
        if is_save:
            plt.savefig(f"plot-delay-emb-{tau=:02d}.png", dpi=600, format="png")
        else:
            plt.show()


def plot_delay_emb(df, col1, col2, tau, ax):
    """
    Plots the 2D delay embedding of two time series from a DataFrame.

    This function creates a scatter plot of two time series, df[lab+ch1] vs.
    df[lab+ch2], with a time delay 'tau' applied to the first series.
    Both the points and the lines connecting them are colored with a
    gradient that represents the progression of time.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        lab (str): The base label for the columns (e.g., 'pos').
        ch1 (str): The identifier for the first channel/variable.
        ch2 (str): The identifier for the second channel/variable.
        tau (int): The time delay (lag) in number of time steps.
                   If tau is 0, no delay is applied.
        ax (matplotlib.axes.Axes): The matplotlib axes object to plot on.
    """
    # --- 1. Data Preparation ---
    # Handle the time delay (tau)
    if tau > 0:
        # s1 is the delayed series
        s1 = df[col1].iloc[:-tau].values
        # s2 is the reference series, shifted to align with s1
        s2 = df[col2].iloc[tau:].values
        str_delay = f" (τ = {tau})"
    else:
        # No delay
        s1 = df[col1].values
        s2 = df[col2].values
        str_delay = " (no delay)"

    # Ensure both series have the same length after slicing
    min_len = min(len(s1), len(s2))
    s1 = s1[:min_len]
    s2 = s2[:min_len]

    points = np.array([s1, s2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # --- 2. Colormap and Normalization ---
    # Create a continuous colormap (e.g., 'viridis', 'plasma', 'cividis')
    cmap = plt.get_cmap("viridis")
    # Create a normalization object to map time steps (0 to N-1) to the colormap (0 to 1)
    norm = plt.Normalize(0, len(s1))  # pyright: ignore

    # --- 3. Plotting ---
    # Create a LineCollection object. The colors are determined by the index of each segment.
    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.5)  # pyright: ignore
    # Set the colors array for the lines. We color based on the start point of each segment.
    lc.set_array(np.arange(len(s1)))
    lc.set_linewidth(1.5)
    ax.add_collection(lc)

    # Plot the points (scatter plot) with the same colormap
    # The color of each point is determined by its time index
    ax.scatter(
        s1, s2, c=np.arange(len(s1)), cmap=cmap, norm=norm, s=3, alpha=0.5, zorder=3
    )

    # --- 4. Aesthetics and Labels ---
    ax.set_title(f"{col1}:{col2}{str_delay}", fontsize=10)
    ax.set_xlabel(f"{col1}" + (f" (t-{tau})" if tau > 0 else " (t)"), fontsize=8)
    ax.set_ylabel(f"{col2} (t)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axis("equal")


if __name__ == "__main__":
    st = 250000
    le = 24000

    new_le = 4000
    st = st + le // 2 - new_le // 2
    le = new_le

    all_df = aa.build_df(st, le)

    # ch = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    # basic_plots(all_df, ch)

    # ch = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    # more_plots(all_df, ch)

    # tau_range = list(range(0, 17, 1))
    # simplex_plots(all_df, tau_range, False)

    col = "vel1"
    train_lib = [1, le // 2]
    pred_lib = [le // 2 + 1, le]
    e = 12
    tp = 12
    tau = 2
    r = e * tau
    simplex = pyEDM.Simplex(
        dataFrame=all_df,
        columns=col,
        target=col,
        lib=train_lib,  # pyright: ignore
        pred=pred_lib,  # pyright: ignore
        E=e,
        Tp=tp,
        tau=-tau,
        exclusionRadius=r,
    )

    plt.figure()
    plt.plot(all_df[col].values, label=col)
    plt.plot(
        range(pred_lib[0] - 1, pred_lib[1] + tp),
        simplex["Observations"].values,  # pyright: ignore
        label="Observation",
    )
    plt.plot(
        range(pred_lib[0] - 1, pred_lib[1] + tp),
        simplex["Predictions"].values,  # pyright: ignore
        label="Prediction",
    )
    plt.legend()
    plt.show()

    valid_predictions = simplex.dropna()  # pyright: ignore
    obs = valid_predictions["Observations"]
    pred = valid_predictions["Predictions"]
    rho = obs.corr(pred)  # pyright: ignore
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    print(f"Prediction skill (rho): {rho:.4f}")
    print(f"RMSE: {rmse:.4f}")
