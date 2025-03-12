#!/usr/bin/env python3

import numpy as np
from filter import *

process_noise_cov=[[1e-5, 0, 0, 0, 0, 0],
                [0, 1e-5, 0, 0, 0, 0],
                [0, 0, 1e-5, 0, 0, 0],
                [0, 0, 0, 1e-3, 0, 0],
                [0, 0, 0, 0, 1e-3, 0],
                [0, 0, 0, 0, 0, 1e-3]]

measurement_noise_cov = np.eye(6)/100

initial_state=[0, 0, 0, 0, 0, 0]

initial_covariance=[[1e-3, 0, 0, 0, 0, 0],
                [0, 1e-3, 0, 0, 0, 0],
                [0, 0, 1e-3, 0, 0, 0],
                [0, 0, 0, 1e-1, 0, 0],
                [0, 0, 0, 0, 1e-1, 0],
                [0, 0, 0, 0, 0, 1e-1]]

dt = 0.2



ekf = EKF(process_noise_cov=process_noise_cov, initial_state=initial_state, initial_covariance=initial_covariance, dt=dt)

aux = True
arr = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0], [4.0, 4.0, 0.0], [5.0, 5.0, 0.0]]
T = 0

while(aux):
    ekf.predict()
    if T%5==0:
        ekf.update(np.asarray(arr[int(T/5)]), dynamic_R=measurement_noise_cov)
        print('----- ',np.asarray(arr[int(T/5)]))
    T += 1
    curr_state = ekf.get_state()
    print("Time: " + str(T) + ", EKF_STATE: " + str(curr_state))