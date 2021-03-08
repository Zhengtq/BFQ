#!/usr/bin/env python
#coding: utf-8


import os
import sys
sys.path.append('../build/python/')
import bfq
import cv2


bfq = bfq.BFQ()

now_img = cv2.imread('./80x80.bmp')

for ind in range(100):
    bfq.Run(now_img)
    final_result = bfq.get_result()
    print(final_result)














