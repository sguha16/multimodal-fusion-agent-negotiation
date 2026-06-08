# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:26:50 2026

@author: sanhi
"""
#the entire dataset should be downloaded here: https://drive.google.com/drive/folders/16AhLPbF4_XiYhNBDa0kL1cOb9vYOF988?hl=fr
#for initial test: a recording was loaded from path:https://drive.google.com/drive/folders/1ur9Dgr7fjE4v0QbdBKXLtk5t8ey__3wn?hl=fr
#-------DATA STRUCTURE DETAILS---------#
#########################################
#Each recording has multiple frames (video available)
#Each frame has 4 chips
#each chip has RX antennas
#each antenna receives 256 chirps
#each chirp has 512 samples, each sample is a+ib (complex no)
#each complex no is of size 2+2= 4bytes
#########################################
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# mkl_fft (Intel's FFT library) and numpy both ship their own copy of OpenMP (a parallel processing library). When Python loads both, it crashes because two copies of the same library conflict.
#=TRUE tells OpenMP "I know there are two copies, don't crash, just pick one." It's a known Anaconda issue

ROOT = r"C:\Users\sanhi\Downloads\RECORD@2020-11-21_11.54.31"
PREFIX = "RECORD@2020-11-21_11.54.31"

# 1. Radar raw ADC — check shape
for ch in range(4):
    path = os.path.join(ROOT, f"{PREFIX}_radar_ch{ch}.bin")
    size_mb = os.path.getsize(path) / 1e6
    print(f"radar_ch{ch}: {size_mb:.0f} MB")
# 2. Events log — see timestamps
log_path = os.path.join(ROOT, f"{PREFIX}_events_log.rec")
with open(log_path, 'r') as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i > 15:
            break
# 3. GPS
gps_path = os.path.join(ROOT, f"{PREFIX}_gps.ascii")
with open(gps_path, 'r') as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i > 5:
            break
#----READ REC USING GITHUB SCRIPT RADIAL--------
##############################################
import sys
sys.path.append(r"C:\Users\sanhi\RADIal_code")
from DBReader.DBReader import SyncReader
from SignalProcessing.rpl import RadarSignalProcessing

SEQ = r"C:\Users\sanhi\Downloads\RECORD@2020-11-21_11.54.31"
CALIB = r"C:\Users\sanhi\RADIal_code\SignalProcessing\CalibrationTable.npy"

# Read one frame
reader = SyncReader(SEQ, tolerance=20000)
sample = next(iter(reader))
# Get the 4 ADC arrays
adc0 = sample['radar_ch0']['data']
adc1 = sample['radar_ch1']['data']
adc2 = sample['radar_ch2']['data']
adc3 = sample['radar_ch3']['data']
print(f"ADC shapes: {adc0.shape}, {adc1.shape}, {adc2.shape}, {adc3.shape}")

# EARLY: Range-Doppler spectrum
rsp_rd = RadarSignalProcessing(CALIB, method='RD', device='cpu', lib='CuPy')
rd = rsp_rd.run(adc0, adc1, adc2, adc3)
print(f"Range-Doppler shape: {rd.shape}")

# MID: Point Cloud
rsp_pc = RadarSignalProcessing(CALIB, method='PC', device='cpu', lib='CuPy')
pc = rsp_pc.run(adc0, adc1, adc2, adc3)
print(f"Point Cloud shape: {pc.shape}  (columns: range, doppler, azimuth, elevation)")

# LATE: same point cloud → cluster into objects (next step)
print(f"\nFirst 5 detections:\n{pc[:5]}")

#CAMERA-raw RGB data
cam = sample['camera']['data']
print(f"Camera shape: {cam.shape}")#1080,1920,3 (pixels.ISP internal)
print(f"Camera dtype: {cam.dtype}")
#visualize 
import matplotlib.pyplot as plt
plt.imshow(cam)
plt.title("Camera Frame")
plt.show()