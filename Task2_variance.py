import numpy as np
import pandas as pd


accel_data = pd.read_csv("Data/0-steady-state_accel.csv")
wrench_data = pd.read_csv("Data/0-steady-state_wrench.csv")


accel_data[["ax", "ay", "az"]] = accel_data[["ax", "ay", "az"]] * 9.81


R_fa = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])


accel_rotated = accel_data[["ax", "ay", "az"]].dot(R_fa.T)


accel_rotated.columns = ["ax_rotated", "ay_rotated", "az_rotated"]


ax_variance = accel_rotated["ax_rotated"].var() * 100
ay_variance = accel_rotated["ay_rotated"].var() * 100
az_variance = accel_rotated["az_rotated"].var() * 100


fx_variance = wrench_data["fx"].var() * 250
fy_variance = wrench_data["fy"].var() * 250
fz_variance = wrench_data["fz"].var() * 250

tx_variance = wrench_data["tx"].var() * 5000
ty_variance = wrench_data["ty"].var() * 5000
tz_variance = wrench_data["tz"].var() * 5000


print(
    f"Accel Variances: ax' = {ax_variance:.4f}, ay' = {ay_variance:.4f}, az' = {az_variance:.4f}"
)
print(
    f"Force Variances: fx = {fx_variance:.4f}, fy = {fy_variance:.4f}, fz = {fz_variance:.4f}"
)
print(
    f"Torque Variances: tx = {tx_variance:.4f}, ty = {ty_variance:.4f}, tz = {tz_variance:.4f}"
)
