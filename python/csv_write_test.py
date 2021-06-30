# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:40:40 2021

@author: kianw
"""

import csv

with open('myfile.csv','a',newline='') as myfile:
    wrtr = csv.writer(myfile, delimiter=',')
    rows = [10,20,30]
    for row in rows:
        wrtr.writerow([123,37.5])
        myfile.flush() # whenever you want