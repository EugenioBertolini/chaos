import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyEDM


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

tau = 4
tp = 12

lib_start = 1
lib_end = le // 2
pred_start = lib_end + 1
pred_end = le

target_signal = "vel1"
final_signals_for_smap = ["pos0", "vel2", "vel11"]

# # Run S-Map for each identified signal against the target
# print(
#     f"\nRunning S-Map: '{final_signals_for_smap}' (predictors) smap '{target_signal}' (target)..."
# )
#
# # SMap returns a dataframe with results
# smap_output = pyEDM.SMap(
#     dataFrame=all_df,
#     columns=[target_signal] + final_signals_for_smap,  # pyright: ignore
#     target=target_signal,
#     lib=f"{lib_start} {lib_end}",
#     pred=f"{pred_start} {pred_end}",
#     E=len(final_signals_for_smap) + 1,
#     Tp=tp,
#     tau=-tau,
#     theta=0,
#     exclusionRadius=25,
#     embedded=True,
# )
# smap_preds = smap_output["predictions"]  # pyright: ignore
# smap_coeffs = smap_output["coefficients"]  # pyright: ignore
# smap_sv = smap_output["singularValues"]  # pyright: ignore
#
# smap_preds.to_csv(f"smap_predictions_{target_signal}_{le}_tau{tau}_Tp{tp}.csv")
# smap_coeffs.to_csv(f"smap_coefficients_{target_signal}_{le}_tau{tau}_Tp{tp}.csv")
# smap_sv.to_csv(f"smap_singular_values_{target_signal}_{le}_tau{tau}_Tp{tp}.csv")


# print(smap_output)
#
# # The S-Map output contains time-varying Jacobian coefficients.
# # The column name 'd(target)/d(predictor)' shows the influence of the predictor on the target.
#
# jacobian_cols = [
#     f"∂{target_signal}/∂{predictor_signal}"
#     for predictor_signal in final_signals_for_smap
# ]
#
# for jacobian_col in jacobian_cols:
#     if jacobian_col in smap_output["coefficients"].columns:
#         plt.figure(figsize=(12, 5))
#         plt.plot(
#             smap_output["coefficients"]["Time"],
#             smap_output["coefficients"][jacobian_col],
#         )
#         plt.title(f"S-Map Interaction Strength: {jacobian_col}")
#         plt.xlabel("Time Step")
#         plt.ylabel(f"Jacobian: {jacobian_col}")
#         plt.grid(True, linestyle="--", alpha=0.6)
#         plt.show()
#     else:
#         print(f"Could not find Jacobian column '{jacobian_col}' in S-Map output.")
#

preds = pd.read_csv(
    f"smap_predictions_{target_signal}_{le}_tau{tau}_Tp{tp}.csv", index_col=0
)
coeffs = pd.read_csv(
    f"smap_coefficients_{target_signal}_{le}_tau{tau}_Tp{tp}.csv", index_col=0
)
sv = pd.read_csv(
    f"smap_singular_values_{target_signal}_{le}_tau{tau}_Tp{tp}.csv", index_col=0
)

print(f"{list(preds.columns)}\n{preds.shape=}\n")
print(f"{list(coeffs.columns)}\n{coeffs.shape=}\n")
print(f"{list(sv.columns)}\n{sv.shape=}")

# plt.figure(figsize=(12, 5))
# plt.plot(all_df[target_signal].values)  # pyright: ignore
# plt.plot(range(le // 2, le + tp), preds["Observations"])
# plt.plot(range(le // 2, le + tp), preds["Predictions"])
# plt.title(f"S-Map Predictions: {target_signal}")
# plt.tight_layout()
# plt.show()

jacobian_cols = [
    f"∂{target_signal}/∂{predictor_signal}"
    for predictor_signal in [target_signal] + final_signals_for_smap
]

fig, ax = plt.subplots(len(jacobian_cols), 1, figsize=(18, 9))
for jacobian_col, a in zip(jacobian_cols, ax):
    a.plot(
        coeffs["Time"],
        coeffs[jacobian_col],
    )
    a.set_title(f"S-Map Interaction Strength: {jacobian_col}")
    a.set_xlabel("Time Step")
    a.set_ylabel(f"Jacobian: {jacobian_col}")

plt.tight_layout()
plt.show()
