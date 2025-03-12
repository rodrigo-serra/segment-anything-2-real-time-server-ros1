#!/usr/bin/env python3

import numpy as np
from ekf import *


# Define process noise covariance (Q)
process_noise_cov=[[1e-1, 0, 0, 0, 0, 0],
                [0, 1e-1, 0, 0, 0, 0],
                [0, 0, 1e-1, 0, 0, 0],
                [0, 0, 0, 1e-1, 0, 0],
                [0, 0, 0, 0, 1e-1, 0],
                [0, 0, 0, 0, 0, 1e-1]]

# Define measurement noise covariance (R) for 3D measurements
measurement_noise_cov = np.eye(3)/10
# measurement_noise_cov = np.eye(6)/10

# Initial state (x, y, z, dx, dy, dz)
initial_state=[0, 0, 0, 0, 0, 0]

# Initial covariance matrix
initial_covariance=[[1e-1, 0, 0, 0, 0, 0],
                [0, 1e-1, 0, 0, 0, 0],
                [0, 0, 1e-1, 0, 0, 0],
                [0, 0, 0, 1e-1, 0, 0],
                [0, 0, 0, 0, 1e-1, 0],
                [0, 0, 0, 0, 0, 1e-1]]

# Time step
dt = 1.0


# Initialize the Extended Kalman Filter
ekf = EKF(process_noise_cov=process_noise_cov,
          initial_state=initial_state,
          initial_covariance=initial_covariance,
          dt=dt)


# Example trajectory data (3D positions at each timestep)
trajectory = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0], 
              [3.0, 3.0, 0.0], [4.0, 4.0, 0.0], [5.0, 5.0, 0.0]]



# Time variable
T = 0
index = 0
aux = True

# Loop to run the EKF prediction and update
while aux:
    ekf.predict()  # Prediction step
    
    # Update EKF with the current measurement every 5 steps
    if T % 5 == 0:
        index = T // 5  # Calculate the correct index for trajectory
        print('----- Measurement:', np.asarray(trajectory[index]))
    
    # Update step with the measurement and dynamic measurement noise covariance
    ekf.update(np.asarray(trajectory[index]), dynamic_R=measurement_noise_cov)
    
    # Increment time step
    T += 1
    
    # Get and print the current state estimate
    curr_state = ekf.get_state()
    print(f"Time: {T}, EKF State: {curr_state}")