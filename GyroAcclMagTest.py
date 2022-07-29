import csv
import numpy as np
from ahrs.filters import EKF, FAMC, Fourati
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


def quaternion_to_euler_angle_vectorized1(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z

if 1:
    gyr_raw = csv.reader(open("data/nomovelong/Gyroscope.csv"))
    acc_raw = csv.reader(open("data/nomovelong/Accelerometer.csv"))
    mag_raw = csv.reader(open("data/nomovelong/Magnetometer.csv"))
    # gyr_raw = csv.reader(open("data/nomovelong/GyroscopeUncalibrated.csv"))
    # acc_raw = csv.reader(open("data/nomovelong/AccelerometerUncalibrated.csv"))
    # mag_raw = csv.reader(open("data/nomovelong/MagnetometerUncalibrated.csv"))
elif 1:
    gyr_raw = csv.reader(open("data/chairspin/GyroscopeUncalibrated.csv"))
    acc_raw = csv.reader(open("data/chairspin/AccelerometerUncalibrated.csv"))
    mag_raw = csv.reader(open("data/chairspin/MagnetometerUncalibrated.csv"))
elif 1:
    gyr_raw = csv.reader(open("data/chairspin/Gyroscope.csv"))
    acc_raw = csv.reader(open("data/chairspin/Accelerometer.csv"))
    mag_raw = csv.reader(open("data/chairspin/Magnetometer.csv"))

mag_list = []
mag_time = []
for i, x in enumerate(mag_raw):
    if i == 0: continue
    mag_time.append(float(x[0]))
    mag_list.append([float(x[2]), float(x[3]), float(x[4])])
mag_arr = np.array(mag_list)


gyr_time = []
gyr_list = []
for i, x in enumerate(gyr_raw):
    if i == 0: continue
    gyr_time.append(float(x[0]))
    gyr_list.append([float(x[2]), float(x[3]), float(x[4])])


acc_time = []
acc_list = []
for i, x in enumerate(acc_raw):
    if i == 0: continue
    acc_time.append(float(x[0]))
    acc_list.append([float(x[2]), float(x[3]), float(x[4])])

acc_idx = []
gyr_idx = [] 
for n in mag_time:
    closest_acc = min(acc_time, key=lambda x:abs(x-n))
    acc_idx.append(acc_time.index(closest_acc))
    closest_gyr = min(gyr_time, key=lambda x:abs(x-n))
    gyr_idx.append(gyr_time.index(closest_gyr))
    
gyr_list = [gyr_list[x] for x in gyr_idx]
acc_list = [acc_list[x] for x in acc_idx]
gyr_arr = np.array(gyr_list)
acc_arr = np.array(acc_list)

ekf = EKF(gyr=gyr_arr, acc=acc_arr, mag=mag_arr)
fo = Fourati(gyr=gyr_arr, acc=acc_arr, mag=mag_arr)


fig, ax = plt.subplots()
ka_eu = [quaternion_to_euler_angle_vectorized1(x[0], x[1], x[2], x[3]) for x in ekf.Q]
fo_eu = [quaternion_to_euler_angle_vectorized1(x[0], x[1], x[2], x[3]) for x in fo.Q]
# ax.plot(mag_time, [x[0] for x in fo_eu], linewidth=2.0, label="x_fo")
# ax.plot(mag_time, [x[1] for x in fo_eu], linewidth=2.0, label="y_fo")
# ax.plot(mag_time, [x[2] for x in fo_eu], linewidth=2.0, label="z_fo")
ax.plot(mag_time, [x[0] for x in ka_eu], linewidth=2.0, label="x_ka")
ax.plot(mag_time, [x[1] for x in ka_eu], linewidth=2.0, label="y_ka")
ax.plot(mag_time, [x[2] for x in ka_eu], linewidth=2.0, label="z_ka")
# ax.plot(t, alt1, linewidth=2.0)
# ax.plot(t, alt2, linewidth=2.0)
# ax.plot(t, alt2, linewidth=2.0)
plt.legend()
plt.show()
