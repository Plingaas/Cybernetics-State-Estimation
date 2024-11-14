import numpy as np


class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x):
        """
        Initialize the Kalman Filter with system parameters.

        A: State transition matrix
        B: Control input matrix
        H: Observation model matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        P: Initial error covariance (posterior)
        x: Initial state estimate (posterior)
        """
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self, u):
        self.x_prior = self.A @ self.x + self.B @ u

        self.P_prior = self.A @ self.P @ self.A.T + self.Q

        return self.x_prior, self.P_prior

    def correct(self, z):
        """
        z: Measurement vector
        """

        S = self.H @ self.P_prior @ self.H.T + self.R
        K = self.P_prior @ self.H.T @ np.linalg.inv(S)  # Kalman Gain

        self.x = self.x_prior + K @ (z - self.H @ self.x_prior)

        I = np.eye(self.P_prior.shape[0])  # Identity matrix
        self.P = (I - K @ self.H) @ self.P_prior

        return self.x, self.P

'''
A = np.eye(9,9)
B = np.zeros((9, 3))
Q = np.eye(9, 9)
H = np.eye(9)  # Measurement matrix (assuming direct measurement of wrench)
R = np.eye(9) * 0.5  # Measurement noise covariance
P0 = np.eye(9) * 100 # Initial error covariance
x0 = np.zeros((9, 1))  # Initial state estimate (force and torque)

filter = KalmanFilter(A, B, H, Q, R, P0, x0)
'''