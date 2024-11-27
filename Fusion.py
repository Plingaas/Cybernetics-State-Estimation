import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Calibration import *
from Kalman_Filter import *
from Task2_variance import *
from concurrent.futures import ThreadPoolExecutor
import time

class Fusion:
    def __init__(self, accel_data, wrench_data, orientation_data, sampling_rates):
        self.accel_data = accel_data
        self.wrench_data = wrench_data
        self.orientation_data = orientation_data
        self.f_r = sampling_rates['orientation']
        self.f_f = sampling_rates['wrench']
        self.f_a = sampling_rates['accel']

        A = np.eye(9)
        B = np.block([
            [np.eye(3)],
            [np.eye(3)*MASS],
            [COM_SKEW * MASS]
        ])

        self.kf = KalmanFilter(A, B)

    def sortData(self):
        
        data = []
        for i in range(len(self.accel_data)):
            data.append((self.accel_data[i][0], 0, self.accel_data[i][1:]))
        for i in range(len(self.wrench_data)):
            data.append((self.wrench_data[i][0], 1, self.wrench_data[i][1:]))
        for i in range(len(self.orientation_data)):
            data.append((self.orientation_data[i][0], 2, self.orientation_data[i][1:]))
        data.sort(key=lambda x: x[0])
        return data

    def run(self):
        dataset = self.sortData()

        t0 = dataset[0][0]

        raw_counter = 0
        timestamps = []
        raw_wrench = []
        filtered_wrench = []
        estimated_wrench = []

        FREQ_SCALING = self.f_r / (self.f_f + self.f_a)
        MICROSEC_TO_SEC = 1e-6
        SEC_TO_MICROSEC = 1e6
        
        z_c = np.zeros((6, 9))
        u = np.zeros((3,1))
        g_fts_lastK = np.zeros((3, 1))

        last_tka = t0
        last_tkf = t0
        g = -9.81
        for t, id, data in dataset:
            match id:
                case 0: # Accelerometer
                    dt = (t - last_tka) * MICROSEC_TO_SEC
                    last_tka += dt * SEC_TO_MICROSEC
                    self.kf.setQ(dt)
                    self.kf.update_za(data * g) # Scale from g to m/s^2
                    self.kf.predict(u)
                    self.kf.correct()
                
                case 1: # FTS
                    dt = (t - last_tkf) * MICROSEC_TO_SEC
                    last_tkf += dt * SEC_TO_MICROSEC
                    self.kf.setQ(dt)
                    self.kf.update_zf(data)
                    self.kf.predict(u)
                    self.kf.correct()
                
                case 2: # Orientation
                    R_ws = np.array(data).reshape(3,3)
                    g_w = np.array([0, 0, g]).reshape(3,1)                    
                    g_fts_k = R_ws @ g_w # Gravity in fts frame
                    u = (g_fts_k - g_fts_lastK)*FREQ_SCALING
                    g_fts_lastK = g_fts_k
                    continue # Don't add plot data
            
            z_c = np.block([
                [np.eye(3)*-MASS, np.eye(3), np.zeros((3,3))],
                [COM_SKEW * -MASS, np.zeros((3,3)), np.eye(3)]
            ]) @ self.kf.x

            # Append plot data
            timestamps.append((t-t0)*MICROSEC_TO_SEC)
            raw_wrench.append(self.wrench_data[raw_counter, 1:])
            filtered_wrench.append(self.kf.x[3:,:])
            estimated_wrench.append(z_c)
            raw_counter += id == 1 # This is to match the amount estimated wrenches with raw data when plotting

        return timestamps, np.array(raw_wrench), np.array(filtered_wrench), np.array(estimated_wrench)

def computeSamplingRate(times):
    return 1e6/(( times[(len(times)-1)] - times[0] )/len(times))

def runFusion(which):
    # Load the data from CSVs

    files = [
        "1-baseline",
        "2-vibrations",
        "3-vibrations-contact"
    ]
    accel_data = pd.read_csv(f"Data/{files[which]}_accel.csv")
    wrench_data = pd.read_csv(f"Data/{files[which]}_wrench.csv")
    orientation_data = pd.read_csv(f"Data/{files[which]}_orientations.csv")
    
    sampling_rates = {
        "accel": computeSamplingRate(accel_data['t']),
        "wrench": computeSamplingRate(wrench_data['t']),
        "orientation": computeSamplingRate(orientation_data['t']),
    }
    
    accel_data['t'] -= 8416 # Phase shift 8416us
    accel_data[['ax', 'ay', 'az']] -= IMU_BIASES # Unbias
    accel_data[['ax', 'ay', 'az']] = (R_fa @ accel_data[['ax', 'ay', 'az']].to_numpy().T).T # Rotate to fts frame

    wrench_data[['fx','fy','fz','tx','ty','tz']] -= WRENCH_BIASES # Unbias
    
    accel_data = accel_data.to_numpy()
    wrench_data = wrench_data.to_numpy()
    orientation_data = orientation_data.to_numpy()
    
    fusion = Fusion(accel_data, wrench_data, orientation_data, sampling_rates)
    data = fusion.run()
    return data

def plotData(index, t, raw_wrench, filtered_wrench, estimated_wrench):

    # Plot Force in Z-direction
    plt.subplot(3, 2, index)
    plt.plot(t, raw_wrench[:, 2], label=r"$F_3$",        color="blue", linewidth=2)
    plt.plot(t, filtered_wrench[:, 2], label=r"$\hat{x}_6$",  color="orange",         linewidth=2)
    plt.plot(t, estimated_wrench[:, 2], label=r"$z_{c,3}$",        color="seagreen",       linewidth=2)
    plt.title("Force in Z-direction")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()

    # Plot Torque in Y-direction
    plt.subplot(3, 2, index + 1)
    plt.plot(t, raw_wrench[:, 4], label=r"$T_2$",        color="blue",    linewidth=2)
    plt.plot(t, filtered_wrench[:, 4], label=r"$\hat{x}_8$",  color="orange",         linewidth=2)
    plt.plot(t, estimated_wrench[:, 4], label=r"$z_{c,5}$",        color="seagreen",       linewidth=2)
    plt.title("Torque in Y-direction")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.legend()

if __name__ == "__main__":
    plt.figure(figsize=(12,6))

    # ThreadPoolExecutor for fun to compute faster. Runs concurrently.
    params = [(0,), (1,), (2,)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(task, *args) for task, args in zip([runFusion, runFusion, runFusion], params)]
        results = [future.result() for future in futures]

    for i, result in enumerate(results):
        plotData((i*2+1), results[i][0], results[i][1], results[i][2], results[i][3])

    plt.tight_layout()
    plt.show()
