import numpy as np
import pandas as pd

data = pd.read_csv("Data/calibration_accel.csv", delimiter=",")

calibrated_data = pd.DataFrame()

fx = data["fx"]  # index 0-23
fy = data["fy"]
fz = data["fz"]
tx = data["tx"]
ty = data["ty"]
tz = data["tz"]
ax = data["ax"]
ay = data["ay"]
az = data["az"]
r11 = data["r11"]
r12 = data["r12"]
r13 = data["r13"]
r21 = data["r21"]
r22 = data["r22"]
r23 = data["r23"]
r31 = data["r31"]
r32 = data["r32"]
r33 = data["r33"]


def extract_all_orientations():
    orientations = []
    for i in range(len(fy)):  # sets the orientation matrix for each index
        matrix = np.array(
            [
                [r11[i], r12[i], r13[i]],
                [r21[i], r22[i], r23[i]],
                [r31[i], r32[i], r33[i]],
            ]
        )
        orientations.append(matrix)
    orientations = np.array(orientations)
    return orientations


""" def extract_orientation(i):
    matrix = [[r11[i], r12[i], r13[i]],
            [r21[i], r22[i], r23[i]],
            [r31[i], r32[i], r33[i]]]
    orientations.append(matrix) """


def xyz_force_biases(fx, fy, fz):
    force_bias_x = fx.mean()
    force_bias_y = fy.mean()
    force_bias_z = fz.mean()
    print(f"force biases: x: {force_bias_x}  |  y: {force_bias_y} | z: {force_bias_z}")
    return fx - force_bias_x, fy - force_bias_y, fz - force_bias_z


def xyz_torque_biases(tx, ty, tz):
    torque_bias_x = tx.mean()
    torque_bias_y = ty.mean()
    torque_bias_z = tz.mean()
    return tx - torque_bias_x, ty - torque_bias_y, tz - torque_bias_z


def IMU_biases():
    IMU_bias_x = ax.mean()
    IMU_bias_y = ay.mean()
    IMU_bias_z = az.mean()
    return IMU_bias_x, IMU_bias_y, IMU_bias_z


def gravity_vec(orientations):
    g_w = [0, 0, -9.81]
    G = []
    for i in range(len(orientations)):
        g_si = (orientations[i].transpose()) @ g_w
        G.append(g_si)
    return np.array(G)


def force_vec(x, y, z):
    F = np.column_stack((x, y, z))
    return np.array(F)


def estimate_mass(G, F):
    num = np.sum(G.transpose() @ F)
    den = np.sum(G.transpose() @ G)
    m = num / den
    m = np.mean(m)
    return m


def calc_A(G):
    A = []
    for i in range(len(G)):
        Ai = np.array(
            [[0, G[i][2], -G[i][1]], [-G[i][2], 0, G[i][0]], [G[i][1], -G[i][0], 0]]
        )
        A.append(Ai)
    A = np.array(A)
    return A


def calc_T(x, y, z):
    T = np.column_stack((x_T, y_T, z_T))
    return T


""" def estimate_COM(A, m, T):
    a = np.sum(A.transpose() @ A)
    #r = 1/m * (A.transpose() @ A).inverse() @ A.transpose() @ T
    return a """


def estimate_COM(A, m, T):
    At_A = np.zeros((3, 3))
    At_T = np.zeros(3)

    for i in range(len(A)):
        At_A += A[i].T @ A[i]
        At_T += A[i].T @ T[i]

    At_A_inv = np.linalg.inv(At_A)

    r = (1 / m) * (At_A_inv @ At_T)

    return r


x, y, z = xyz_force_biases(fx, fy, fz)
x_T, y_T, z_T = xyz_torque_biases(tx, ty, tz)
orientations = extract_all_orientations()
G = gravity_vec(orientations)
F = force_vec(x, y, z)
m = estimate_mass(G, F)
A = calc_A(G)
T = calc_T(x_T, y_T, z_T)

IMU_BIAS_X = ax.mean()
IMU_BIAS_Y = ay.mean()
IMU_BIAS_Z = az.mean()
IMU_BIASES = [IMU_BIAS_X, IMU_BIAS_Y, IMU_BIAS_Z]

FORCE_BIAS_X = fx.mean()
FORCE_BIAS_Y = fy.mean()
FORCE_BIAS_Z = fz.mean()
TORQUE_BIAS_X = tx.mean()
TORQUE_BIAS_Y = ty.mean()
TORQUE_BIAS_Z = tz.mean()
WRENCH_BIASES = [FORCE_BIAS_X, FORCE_BIAS_Y, FORCE_BIAS_Z, TORQUE_BIAS_X, TORQUE_BIAS_Y, TORQUE_BIAS_Z]
MASS = m
COM_ESTIMATE = estimate_COM(A, m, T)
COM_SKEW = np.array([
    [0, -COM_ESTIMATE[2], COM_ESTIMATE[1]],
    [COM_ESTIMATE[2], 0, COM_ESTIMATE[0]],
    [-COM_ESTIMATE[1], COM_ESTIMATE[0], 0]
])
COM_MAG = np.sqrt(COM_ESTIMATE[0]**2 + COM_ESTIMATE[1]**2 + COM_ESTIMATE[2]**2)