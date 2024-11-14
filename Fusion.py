import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Calibration import *
from Kalman_Filter import *
'''
class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = A  # State transition matrix
        self.B = B  # Control input matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial covariance estimate

    def predict(self, u=None):
        # Prediction step
        self.x = (
            np.dot(self.A, self.x) + np.dot(self.B, u)
            if u is not None
            else np.dot(self.A, self.x)
        )
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

    def update(self, z):
        # Measurement update step
        y = z - np.dot(self.H, self.x)  # Innovation
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Innovation covariance
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))  # Kalman gain
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
'''

class Fusion:
    def __init__(self, accel_data, wrench_data, orientation_data, sampling_rates):
        self.accel_data = accel_data
        self.wrench_data = wrench_data
        self.orientation_data = orientation_data
        self.sampling_rates = sampling_rates  # {sensor_name: rate}

        # State space model matrices (example)
        self.A = np.eye(9)


        self.B = np.zeros((9, 3))
        for y in range(3):
            self.B[y][y] = 1
        for y in range(3):
            self.B[y+3][y] = MASS

        # Skew symmetric r_s multiplied by mass for last 3x3 of self.B as per eq. (16)
        self.B[6, 0] = 0
        self.B[6, 1] = -COM_ESTIMATE[2] * MASS
        self.B[6, 2] = COM_ESTIMATE[1] * MASS
        self.B[7, 0] = COM_ESTIMATE[2] * MASS
        self.B[7, 1] = 0
        self.B[7, 2] = COM_ESTIMATE[0] * MASS
        self.B[8, 0] = -COM_ESTIMATE[1] * MASS
        self.B[8, 1] = COM_ESTIMATE[0] * MASS
        self.B[8, 2] = 0


        self.Q = np.eye(9, 9)
        t_k = 1/self.sampling_rates['orientation'] # Timestep
        sigma_k = 0.5 # Process variance
        COM_MAG = np.sqrt(COM_ESTIMATE[0]**2 + COM_ESTIMATE[1]**2 + COM_ESTIMATE[2]**2)
        print(f"t_k: {t_k}")
        for y in range(3):
            self.Q[y][y] = 1 * t_k * sigma_k
        for y in range(3):
            self.Q[y+3][y+3] = MASS * t_k * sigma_k
        for y in range(3):
            self.Q[y+6][y+6] = MASS * COM_MAG * t_k * sigma_k

        self.H = np.eye(9)  # Measurement matrix (assuming direct measurement of wrench)
        self.R = np.eye(9) * 0.5  # Measurement noise covariance
        self.x0 = np.zeros((9, 1))  # Initial state estimate (force and torque)
        self.P0 = np.eye(9) * 100  # Initial covariance estimate (uncertainty)
        # Initialize Kalman filter
        self.kf = KalmanFilter(self.A, self.B, self.H, self.Q, self.R, self.P0, self.x0)

    def synchronize_data(self):
        """
        Synchronize the data from all sensors based on their timestamps.
        """
        # Interpolation for synchronizing data
        accel_interpolator = interp1d(
            self.accel_data["t"],
            self.accel_data[["ax", "ay", "az"]],
            axis=0,
            fill_value="extrapolate",
        )
        wrench_interpolator = interp1d(
            self.wrench_data["t"],
            self.wrench_data[["fx", "fy", "fz", "tx", "ty", "tz"]],
            axis=0,
            fill_value="extrapolate",
        )
        orientation_interpolator = interp1d(
            self.orientation_data["t"],
            self.orientation_data[
                ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]
            ],
            axis=0,
            fill_value="extrapolate",
        )

        # Use the accelerometer timestamps as the reference
        common_timestamps = self.orientation_data["t"].values

        # Interpolate wrench and orientation data onto the orientation timestamps
        accel_synced = accel_interpolator(common_timestamps)
        wrench_synced = wrench_interpolator(common_timestamps)
        orientation_synced = orientation_interpolator(common_timestamps)

        return common_timestamps, accel_synced, wrench_synced, orientation_synced

    def run(self):
        """
        Run the Kalman filter with synchronized sensor data.
        """
        timestamps, accel_data, wrench_data, orientation_data = self.synchronize_data()
        raw_fx, raw_fy, raw_fz = [], [], []  # Raw force data
        filtered_fx, filtered_fy, filtered_fz = (
            [],
            [],
            [],
        )  # Filtered (estimated) force data
        raw_tx, raw_ty, raw_tz = [], [], []  # Raw force data
        filtered_tx, filtered_ty, filtered_tz = (
            [],
            [],
            [],
        )

        g_s_lastK = np.zeros((3, 1))
        freq_scaling = 100/(700+254)
        for t, accel, wrench, orientation in zip(
            timestamps, accel_data, wrench_data, orientation_data
        ):
            # Combine the sensor data into a measurement vector
            # For example, we'll directly use the FTS data for measurement,
            # as it provides the contact wrench directly.
            # We might also need to use the accelerometer and orientation data in the state prediction step.

            # For the measurement, let's assume the wrench is the primary sensor input.
            measurement = (
                wrench  # Direct measurement from FTS data: [fx, fy, fz, tx, ty, tz]
            )
            # Prediction step of the Kalman filter
            
            R_ws = np.zeros((3, 3))
            R_ws[0,0] = orientation[0]
            R_ws[0,1] = orientation[1]
            R_ws[0,2] = orientation[2]
            R_ws[1,0] = orientation[3]
            R_ws[1,1] = orientation[4]
            R_ws[1,2] = orientation[5]
            R_ws[2,0] = orientation[6]
            R_ws[2,1] = orientation[7]
            R_ws[2,2] = orientation[8]
            R_ws = R_ws.transpose()
            g_w = np.zeros((3,1))
            g_w[2] = -9.81
            
            g_s_k = R_ws @ g_w # Gravity in fts frame
            u = (g_s_k - g_s_lastK)*freq_scaling
            g_s_lastK = g_s_k
            self.kf.predict(u)

            # Update step of the Kalman filter with new measurements
            self.kf.correct(measurement)

            # You can access the current state estimate (contact wrench)
            estimated_state = self.kf.x
            #print(f"Time: {t}, Estimated State: {estimated_state.flatten()}")
            # Store the raw data for comparison
            raw_fx.append(wrench[0])
            raw_fy.append(wrench[1])
            raw_fz.append(wrench[2])
            # Store the filtered (estimated) data
            #print(f"kf: {self.kf.x}")
            filtered_fx.append(self.kf.x[3, 0])
            filtered_fy.append(self.kf.x[4, 0])
            filtered_fz.append(self.kf.x[5, 0])

            raw_tx.append(wrench[3])
            raw_ty.append(wrench[4])
            raw_tz.append(wrench[5])
            filtered_tx.append(self.kf.x[6, 0])
            filtered_ty.append(self.kf.x[7, 0])
            filtered_tz.append(self.kf.x[8, 0])


        # Plotting the unfiltered (raw) vs filtered data
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(timestamps, raw_fz, label="Raw Fz", color="r")
        plt.plot(timestamps, filtered_fz, label="Filtered Fz (Kalman)", color="b")
        plt.title("Force in Z-direction")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(timestamps, raw_ty, label="Raw Ty", color="r")
        plt.plot(timestamps, filtered_ty, label="Filtered Ty (Kalman)", color="b")
        plt.title("Torque in y-direction")
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.legend()

        '''
        # Plot Force in X-direction
        plt.subplot(3, 2, 1)
        plt.plot(timestamps, raw_fx, label="Raw Fx", linestyle="--", color="r")
        plt.plot(timestamps, filtered_fx, label="Filtered Fx (Kalman)", color="b")
        plt.title("Force in X-direction")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.legend()

        # Plot Force in Y-direction
        plt.subplot(3, 2, 3)
        plt.plot(timestamps, raw_fy, label="Raw Fy", linestyle="--", color="r")
        plt.plot(timestamps, filtered_fy, label="Filtered Fy (Kalman)", color="b")
        plt.title("Force in Y-direction")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.legend()

        # Plot Force in Z-direction
        plt.subplot(3, 2, 5)
        plt.plot(timestamps, raw_fz, label="Raw Fz", linestyle="--", color="r")
        plt.plot(timestamps, filtered_fz, label="Filtered Fz (Kalman)", color="b")
        plt.title("Force in Z-direction")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.legend()

         # Plot Force in X-direction
        plt.subplot(3, 2, 2)
        plt.plot(timestamps, raw_tx, label="Raw Tx", linestyle="--", color="r")
        plt.plot(timestamps, filtered_tx, label="Filtered Tx (Kalman)", color="b")
        plt.title("Torque in X-direction")
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.legend()

        # Plot Force in Y-direction
        plt.subplot(3, 2, 4)
        plt.plot(timestamps, raw_ty, label="Raw Ty", linestyle="--", color="r")
        plt.plot(timestamps, filtered_ty, label="Filtered Ty (Kalman)", color="b")
        plt.title("Torque in Y-direction")
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.legend()

        # Plot Force in Z-direction
        plt.subplot(3, 2, 6)
        plt.plot(timestamps, raw_tz, label="Raw Tz", linestyle="--", color="r")
        plt.plot(timestamps, filtered_tz, label="Filtered Tz (Kalman)", color="b")
        plt.title("Torque in Z-direction")
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.legend()
        '''
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Load the data from CSVs

    files = [
        "1-baseline",
        "2-vibrations",
        "3-vibrations-contact"
    ]
    index = 0
    accel_data = pd.read_csv(f"Data/{files[index]}_accel.csv")  # Adjust file names and columns as needed
    # Unbiasing
    accel_data['ax'] -= IMU_BIAS_X
    accel_data['ay'] -= IMU_BIAS_Y
    accel_data['az'] -= IMU_BIAS_Z

    #Scaling acceleration to newtons F = ma
    accel_data['ax'] *= MASS
    accel_data['ay'] *= MASS
    accel_data['az'] *= MASS

    wrench_data = pd.read_csv(f"Data/{files[index]}_wrench.csv")
    #Unbiasing
    wrench_data['fx'] -= FORCE_BIAS_X
    wrench_data['fy'] -= FORCE_BIAS_Y
    wrench_data['fz'] -= FORCE_BIAS_Z
    wrench_data['tx'] -= TORQUE_BIAS_X
    wrench_data['ty'] -= TORQUE_BIAS_Y
    wrench_data['tz'] -= TORQUE_BIAS_Z

    orientation_data = pd.read_csv(f"Data/{files[index]}_orientations.csv")

    # Sampling rates for each sensor (Hz)
    sampling_rates = {
        "accel": 254,  # Example: Accel at 100 Hz
        "wrench": 700,  # Example: Wrench at 50 Hz
        "orientation": 100,  # Example: Orientation at 10 Hz
    }
    fusion = Fusion(accel_data, wrench_data, orientation_data, sampling_rates)
    fusion.run()
