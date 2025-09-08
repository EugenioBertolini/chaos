import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
import pyEDM
import Aa_load_data as aa


def plot_correlations(all_df, col, train_lib, pred_lib, e, maxtp, tau, r, ax, colors):
    acf_ = acf(all_df[col], nlags=maxtp)
    pred_interval_df = pyEDM.PredictInterval(
        dataFrame=all_df,
        columns=col,
        target=col,
        lib=train_lib,  # pyright: ignore
        pred=pred_lib,  # pyright: ignore
        maxTp=maxtp,
        E=e,
        tau=-tau,
        exclusionRadius=r,
        numProcess=32,
        showPlot=False,
    )
    ax.plot(np.abs(acf_), "o-", label=f"corr {col}", color=colors[0])
    ax.plot(
        pred_interval_df["Tp"],
        pred_interval_df["rho"],
        "--",
        label=f"pred interval {col}",
        color=colors[1],
    )


def plot_all_correlations(all_df, ch, train_lib, pred_lib, e, maxtp, tau, r):
    _, ax = plt.subplots(3, 2, figsize=(18, 9))
    c = sns.color_palette("Paired")
    for ch_i, ax_i in zip(ch, ax.flatten()):
        print(f"{ch_i=}")
        plot_correlations(
            all_df,
            f"pos{ch_i[0]}",
            train_lib,
            pred_lib,
            e,
            maxtp,
            tau,
            r,
            ax_i,
            c[:2],
        )
        plot_correlations(
            all_df,
            f"pos{ch_i[1]}",
            train_lib,
            pred_lib,
            e,
            maxtp,
            tau,
            r,
            ax_i,
            c[2:4],
        )
        plot_correlations(
            all_df,
            f"vel{ch_i[0]}",
            train_lib,
            pred_lib,
            e,
            maxtp,
            tau,
            r,
            ax_i,
            c[4:6],
        )
        plot_correlations(
            all_df,
            f"vel{ch_i[1]}",
            train_lib,
            pred_lib,
            e,
            maxtp,
            tau,
            r,
            ax_i,
            c[6:8],
        )
        plot_correlations(
            all_df,
            f"acc{ch_i[0]}",
            train_lib,
            pred_lib,
            e,
            maxtp,
            tau,
            r,
            ax_i,
            c[8:10],
        )
        plot_correlations(
            all_df,
            f"acc{ch_i[1]}",
            train_lib,
            pred_lib,
            e,
            maxtp,
            tau,
            r,
            ax_i,
            c[10:],
        )

        ax_i.set_xlim(0, maxtp)
        ax_i.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_simplex(all_df, col, train_lib, pred_lib, e, tp, tau, r):
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

    valid_predictions = simplex.dropna()  # pyright: ignore
    obs = valid_predictions["Observations"]
    pred = valid_predictions["Predictions"]
    rho = obs.corr(pred)  # pyright: ignore
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    print(f"Prediction skill (rho): {rho:.4f}")
    print(f"RMSE: {rmse:.4f}")

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


def plot_e_tau_landscape(
    all_df, train_lib, pred_lib, max_e, tp_range, tau_range, is_save=False
):
    emb_df = None
    for col in all_df.columns:
        for tau in tau_range:
            r = max_e * tau
            print(f"{col=}, {tau=}")
            tmp_df = None
            for tp in tp_range:
                tmp_tp_df = pyEDM.EmbedDimension(
                    dataFrame=all_df,
                    columns=col,
                    target=col,
                    lib=train_lib,  # pyright: ignore
                    pred=pred_lib,  # pyright: ignore
                    maxE=max_e,
                    Tp=tp,
                    tau=-tau,
                    exclusionRadius=r,
                    showPlot=False,
                    numProcess=32,
                )
                if tmp_df is None:
                    tmp_df = tmp_tp_df
                else:
                    tmp_df += tmp_tp_df

            tmp_df /= len(tp_range)  # pyright: ignore
            rho_col_name = f"rho_{col}_tau{tau}"
            tmp_df.rename(columns={"rho": rho_col_name}, inplace=True)
            if emb_df is None:
                emb_df = tmp_df
            else:
                emb_df = pd.merge(
                    emb_df, tmp_df[["E", rho_col_name]], on="E", how="outer"
                )

    if is_save:
        emb_df.to_csv(  # pyright: ignore
            f"e_tau_landscape_{pred_lib[1]}_Emax{max_e}_taumax{tau_range[-1]}.csv",
        )

    plot_rho_colormaps_by_col(emb_df, max_e, -np.asarray(tau_range))


if __name__ == "__main__":
    st = 250000
    le = 24000

    new_le = 4000
    st = st + le // 2 - new_le // 2
    le = new_le

    all_df = aa.build_df(st, le)

    # ch = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    # train_lib = [1, le // 2]
    # pred_lib = [le // 2 + 1, le]
    # e = 3
    # maxtp = 20
    # tau = 6
    # r = e * tau
    # plot_all_correlations(all_df, ch, train_lib, pred_lib, e, maxtp, tau, r)

    # col = "vel1"
    # train_lib = [1, le // 2]
    # pred_lib = [le // 2 + 1, le]
    # e = 3
    # tp = 12
    # tau = 6
    # r = e * tau
    # plot_simplex(all_df, col, train_lib, pred_lib, e, tp, tau, r)

    # train_lib = [1, le // 2]
    # pred_lib = [le // 2 + 1, le]
    # max_e = 12
    # tp_range = list(range(3, 18, 3))
    # tau_range = list(range(1, 19))
    # plot_e_tau_landscape(all_df, train_lib, pred_lib, max_e, tp_range, tau_range, True)

    max_e = 12
    max_tau = 18
    emb_df = pd.read_csv(
        f"e_tau_landscape_{le}_Emax{max_e}_taumax{max_tau}.csv", index_col=0
    ).drop(columns=["E"])
    emb_tensor = emb_df.to_numpy().reshape(len(all_df.columns), max_e, -1)
    for i, emb_mat in enumerate(emb_tensor):
        print(
            f"{all_df.columns[i]} -> E:",
            np.argmax(emb_mat.max(1)) + 1,
            "tau:",
            max_tau - np.argmax(emb_mat.max(0)),
        )
