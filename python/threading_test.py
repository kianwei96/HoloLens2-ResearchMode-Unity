# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:28:46 2021

@author: kianw
"""

import threading
import time
import cv2
import numpy as np

def img_processor():
    time_start = time.time()
    global img_queue, experiment_duration
    while True:
        if len(img_queue) > 0:
            data = img_queue.pop()
            cv2.imwrite("data/" + str(time.time()) + "_ab.tiff", data)
            cv2.imwrite("data/" + str(time.time()) + "_dep.tiff", data)
        if time.time() - time_start > experiment_duration + 3:
            break
    
def stream_listener():
    count = 0
    time_start = time.time()
    global img_queue, experiment_duration
    while True:
        img_queue.append(np.random.randint(255, size=(512,512)))
        count += 1
        if time.time() - time_start > experiment_duration:
            print(count)
            break
    
def multithreads():
    shutdown_timer = time.time()
    thread1 = threading.Thread(target=stream_listener(), args=())
    thread2 = threading.Thread(target=img_processor(), args=())
    thread1.start()
    thread2.start()
    
experiment_duration = 3
img_queue = []

multithreads()
