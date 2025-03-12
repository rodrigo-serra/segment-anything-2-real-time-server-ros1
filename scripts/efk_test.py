#!/usr/bin/env python3

import numpy as np
from filter import *

process_noise_cov=[[1e-1, 0, 0, 0, 0, 0],
                [0, 1e-1, 0, 0, 0, 0],
                [0, 0, 1e-1, 0, 0, 0],
                [0, 0, 0, 1e-1, 0, 0],
                [0, 0, 0, 0, 1e-1, 0],
                [0, 0, 0, 0, 0, 1e-1]]

measurement_noise_cov = np.eye(3)/10
# measurement_noise_cov = np.eye(6)/10

initial_state=[0, 0, 0, 0, 0, 0]

initial_covariance=[[1e-1, 0, 0, 0, 0, 0],
                [0, 1e-1, 0, 0, 0, 0],
                [0, 0, 1e-1, 0, 0, 0],
                [0, 0, 0, 1e-1, 0, 0],
                [0, 0, 0, 0, 1e-1, 0],
                [0, 0, 0, 0, 0, 1e-1]]

dt = 1.0



ekf = EKF(process_noise_cov=process_noise_cov, initial_state=initial_state, initial_covariance=initial_covariance, dt=dt)

aux = True
arr = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0], [4.0, 4.0, 0.0], [5.0, 5.0, 0.0]]
T = 0
index = 0
while(aux):
    ekf.predict()
    if T%5==0:
        index = int(T/5)
        print('----- ',np.asarray(arr[int(T/5)]))
    ekf.update(np.asarray(arr[index]), dynamic_R=measurement_noise_cov)
    T += 1
    curr_state = ekf.get_state()
    print("Time: " + str(T) + ", EKF_STATE: " + str(curr_state))