# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:36:49 2021

@author: kianw
"""

import time
import cv2
import numpy as np
import os

def stream_listener():
    count = 0
    global experiment_duration
    time_start = time.time()
    while True:
        data = np.random.randint(255, size=(512,512))
        cv2.imwrite("data/" + str(time.time()) + "_ab.tiff", data)
        cv2.imwrite("data/" + str(time.time()) + "_dep.tiff", data)
        count += 1
        if time.time() - time_start > experiment_duration:
            print(count)
            break
 
experiment_duration = 3
print(os.getcwd())
stream_listener()