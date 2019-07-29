# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:58:31 2019

@author: HP
"""

from scipy.io.wavfile import read
import numpy as npy
import matplotlib.pyplot as plt
import sys
import sounddevice as sd
from scipy.signal import find_peaks
from scipy.misc import electrocardiogram 

npy.set_printoptions(threshold= sys.maxsize)

#read audio files
rate, data = read("bit.wav")

data = npy.array(data)

data_m = npy.mean(data,1)

#720 samples in 15ms

sd.play(data[30000:50000], rate) #30-51

#x = electrocardiogram()[2000:4000]
#peaks,_ = find_peaks(x, height = 1)
#plt.plot(x)
#plt.plot(peaks, x[peaks], "x")
#print (peaks)

plt.plot(data_m[30000:50000])