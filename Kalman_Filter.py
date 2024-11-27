import numpy as np
from Task2_variance import *
from Calibration import *

class KalmanFilter:
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.H = None
        self.Q = None
        self.R = None
        self.P = np.eye(9)
        self.x = np.zeros((9,1))
        self.P_prior = np.zeros((9,9))
        self.x_prior = np.zeros((9,1))
        self.x[7,0] = -0.4
        self.z = None
        self.H_f = np.block([
            [np.zeros((3,3)), np.eye(3), np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)]
        ])
        self.H_a = np.block([
            [np.eye(3), np.zeros((3,3)), np.zeros((3,3))]
        ])

        self.R_f = np.diag([fx_variance, fy_variance, fz_variance, tx_variance, ty_variance, tz_variance])
        self.R_a = np.diag([ax_variance, ay_variance, az_variance])

    def predict(self, u):
        self.x_prior = self.A @ self.x + self.B @ u

        self.P_prior = self.A @ self.P @ self.A.T + self.Q

        return self.x_prior, self.P_prior

    def correct(self): # update
        S = self.H @ self.P_prior @ self.H.T + self.R
        K = self.P_prior @ self.H.T @ np.linalg.inv(S)  # Kalman Gain

        self.x = self.x_prior + K @ (self.z - self.H @ self.x_prior)

        self.P = (np.eye(9) - K @ self.H) @ self.P_prior

        return self.x, self.P
    
    def update_zf(self, x):
        self.H = self.H_f
        self.R = self.R_f
        z_f = self.H @ x
        self.z = z_f

    def update_za(self, x):
        self.H = self.H_a
        self.R = self.R_a 
        z_a = self.H @ x
        self.z = z_a
    
    def setQ(self, delta_t_k):
        sigmaK = 0.5
        self.Q = np.diag([1, 1, 1, MASS, MASS, MASS, MASS*COM_MAG, MASS*COM_MAG, MASS*COM_MAG]) * sigmaK * delta_t_k