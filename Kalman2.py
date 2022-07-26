"""
Oh shiet, here we go again
"""
import numpy as np
from filterpy.common import Q_discrete_white_noise
import math
import numpy as np
from numpy.random import randn

def compute_dog_data(z_var, process_var, count=1, dt=1.):
    "returns track, measurements 1D ndarrays"
    x, vel = 0., 1.
    z_std = math.sqrt(z_var) 
    p_std = math.sqrt(process_var)
    xs, zs = [], []
    for _ in range(count):
        v = vel + (randn() * p_std)
        x += v*dt        
        xs.append(x)
        zs.append(x + randn() * z_std)        
    return np.array(xs), np.array(zs)

def my_update(z, x, P, R, H):

    S = np.dot(H, np.dot(P, H.T)) + R

    K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))

    y = z - np.dot(H, x)

    x += np.dot(K, y)
    P = P - np.dot(K, np.dot(H, P))

    return x, P

def my_predict(x, P, F, Q):

    x = np.dot(F, x)
    P = np.dot(F, np.dot(P, F.T)) + Q

    return x, P


if __name__ == "__main__":

    R_var = 10.
    Q_var = 0.01

    # initial state and covariance
    x   = np.array([[10], [4.5]])
    P   = np.array([[500., 0.], [0., 49.]])
    
    
    # prediction
    dt  = 1.
    F   = np.array([[1., dt], [0., 1.]])

    # revert back to measurement
    H   = np.array([[1., 0.]])

    # measurement uncertainty
    R = [[R_var]]

    # process noise
    Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)

    count = 5
    track, zs = compute_dog_data(R_var, Q_var, count)

    xs, cov = [], []

    for z in zs:
        x, P = my_predict(x, P, F, Q)

        x, P = my_update(z, x, P, R, H)

        xs.append(x)
        cov.append(P)


    x   = np.array([[10], [4.5]])
    P   = np.array([[500., 0.], [0., 49.]])
    xs2, cov2 = [], []
    for z in zs:
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        
        #update
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        y = z - H @ x

        x += K @ y
        P = P - K @ H @ P
        
        xs2.append(x)
        cov2.append(P)
    
    print(xs)
    print(xs2)
    print("\n")
    print(cov)
    print(cov2)
    


