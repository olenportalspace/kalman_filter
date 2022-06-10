
import numpy as np
from filterpy.common import Q_discrete_white_noise

import MyMath as mm

"""
My first attempt at implementing a Kalman Filter in Python




"""

def reshape_z(z, dim_z, ndim):
    """ ensure z is a (dim_z, 1) shaped vector"""

    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        raise ValueError('z must be convertible to shape ({}, 1)'.format(dim_z))

    if ndim == 1:
        z = z[:, 0]

    if ndim == 0:
        z = z[0, 0]

    return z

def update2(x, P, z, R, H=None, return_all=False):
    if z is None:
        if return_all:
            return x, P, None, None, None, None
        return x, P

    if H is None:
        H = np.array([1])

    if np.isscalar(H):
        H = np.array([H])

    Hx = np.atleast_1d(np.dot(H, x))
    z = reshape_z(z, Hx.shape[0], x.ndim)

    # error (residual) between measurement and prediction
    y = z - Hx

    print(f"Residual Y: {y}")

    # project system uncertainty into measurement space
    S = np.dot(np.dot(H, P), H.T) + R



    print(f"x: {x}\nP: {P}\nz: {z}\nR: {R}\nH: {H}\n\n")

    print(f"\nH.T:\n{H.T}\nH:\n{H}\nP:\n{P}\nP@H.T:\n{np.dot(P, H.T)}")

    # map system uncertainty into kalman gain
    try:
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    except:
        # can't invert a 1D array, annoyingly
        K = np.dot(np.atleast_1d(np.dot(P, H.T)), 1./S)


    # predict new x with residual scaled by the kalman gain
    x = x + np.dot(K, y)

    # P = (I-KH)P(I-KH)' + KRK'
    KH = np.dot(K, H)

    try:
        I_KH = np.eye(KH.shape[0]) - KH
    except:
        I_KH = np.array([1 - KH])
    P = np.dot(np.dot(I_KH, P), I_KH.T) + np.dot(np.dot(K, R), K.T)


    return x, P

    

    



def predict(x, P, F, Q):
    """
    Variables predict:
        x, P    are the state mean and covariance.
        F       is the state transition function.
        Q       is the process covariance matrix
        B, u    Models the control input
    

    x_hat = F(x) + B(u)
    P_hat = FPF.T + Q

    Not yet learned 
    """

    x_hat = np.dot(F.copy(), x.copy())
    P_hat = np.dot(F.copy(), np.dot(P.copy(), mm.my_transpose(F.copy())))
    P_hat = [[P_hat[i][j] + Q[i][j] for j in range(len(P_hat[0]))]for i in range(len(P_hat))]


    return x_hat, P_hat


def update(x, P, z, R, H=None):
    """
    Variables update:
        H       is the measurement function
        z, R    are the measurement mean and noise covariance
        y, K    residual and Kalman gain


    """
    if z is None:
        return x, P

    if H is None:
        H = np.array([H])

    if np.isscalar(H):
        H = np.array([H])
    
    Hx = np.atleast_1d(np.dot(H, x))
    
    
    z = np.atleast_2d(z)
    if z.shape[1] == Hx.shape[0]:
        z = z.T
        print("First")

    if z.shape != (Hx.shape[0], 1):
        raise ValueError('z must be convertible to shape ({}, 1)'.format(Hx.shape[0]))

    if x.ndim == 1:
        z = z[:, 0]
        print("Second")

    if x.ndim == 0:
        z = z[0, 0]
        print("Third")

    y = z - Hx
    print(f"Residual Y: {y}")
    return 1, 2




if __name__ == "__main__":
    """
    
    """
    # Design state variable
    """
    Best to choose this based on the first sensor readings or an actual
    known position and velocity from somewhere. 
    """
    x = np.array([10.0, 4.5])
    print(f"State variable x:\n{x} \n")
    # ---------------------------------------

    # ---------------------------------------
    # Design state covariance
    """
    500 is just chosen because we are quite uncertain about actual starting
    position. Here 500 is the variance for the position, aka the first value
    in x

    49 is chosen as the top speed for a dog is 21m/s. It is known that 99,7%
    of possible values will fall within 3 standard deviations from the mean.
    21 * 3 = 49
    """
    P = [[500., 0.],
         [0., 49.]]
    print(f"State covariance P:\n{P} \n")
    # ---------------------------------------

    # ---------------------------------------
    # Design the process model
    """
    A mathematical model which describes the behavior of the system.
    Used to predict the state of the system after a discrete time step.

    also initialize the timestep as it is used in the state transition
    matrix - later to be  a function
    """
    dt = 0.1
    F = [[1, dt],
         [0, 1]]
    
    print(f"Process model F:\n{F} \n")

    # ---------------------------------------

    # ---------------------------------------
    # Design process noise
    """
    Process noise is noise added as there are alot of different things
    affecting how the movement behaves.

    For example, wind can greatly affect how far an object can move.

    Will go further in depth in Kalman Math chapter
    """
    Q = Q_discrete_white_noise(dim=2, dt=1., var=2.35)

    print(f"Process noise Q:\n{Q} \n")
    
    # ---------------------------------------

    # ---------------------------------------
    # Design the control function
    """
    B is the control input model or control function
    u is the input itself, for example a voltage or something similar

    More in later chapters
    """

    B = 0. # my dog doesn't listen to me!
    u = 0
    # ---------------------------------------


    # ---------------------------------------
    # Design the measurement function
    """
    A KF computes the update step in measurement space.

    This means that we will go from state back to value from measurement

    For example from celsius back to volts if there is a temperature
    sensor.

    Then we can use that to compute the residual y
    y = z - (CELSIUS_TO_VOLTS * x)

    More generalized:
    y = z - H(x_hat)
    """
    H = [[1., 0.]]
    # ---------------------------------------


    # ---------------------------------------
    # Design the measurement
    """
    z is the measurement(s)
    z = [z1, z2, ..., zn]

    for one measurement it is: [z]
    
    measurement noise matrix might not be a pure gaussian.
    The variance can be skewed depending on temperature for example.
    Those problems we will deal with later

    [[1., 0., 0.],
     [0., 2., 0.],
     [0., 0., n.]]
    
    for one sensor we just have
    [n.]
    """
    R = np.array([[5.]])
    print(f"Measurement function R:\n{R} \n")


    x, P = predict(x, P, F, Q)
    print(f"x = {x}")
    print(f"P = {P}")

    z = 1.
    print(f"x: {x}\nP: {P}\nz: {z}\nR: {R}\nH: {H}\n\n")


    new_x, new_P = update(x.copy(), np.array(P.copy()), z, np.array(R.copy()), np.array(H.copy()))
    new_x2, new_P2 = update2(x.copy(), np.array(P.copy()), z, np.array(R.copy()), np.array(H.copy()))

    print("\nCopied version:")
    print(f"x_copy = \t{new_x}\nx_original = \t{new_x2}\n")
    print(f"P_copy = \n{new_P}\n\nP_original = \n{new_P2}")

