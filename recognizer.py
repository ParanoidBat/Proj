from scipy.io.wavfile import read
import numpy as npy
import os
import glob
from scipy import signal

def energyFrames(time, spectogram):
    average =[]
    
    i = 0
    while i < time.size:
        if i+5 >= time.size: break
        #take mean of matrix along x-axis
        vector = npy.mean(spectogram[:170,i:i+5],1) #frequencies after index 170 are useless. As 8k Hz is the last formant of use
        
        #get a single value out of the vector, which represents a single value energy magnitude of the frame
        vector_avg = npy.mean(vector)
        average.append(vector_avg)
        
        i+=5
    
    return average

def plotSpec(audio, rate):
    freq, time, sx = signal.spectrogram(audio, fs=rate, window="hamming", nperseg=1024, noverlap=924, detrend=False, scaling="spectrum")
    
    return freq, time, sx



path = "Samples/"
audio_lengths = []

for filename in glob.glob(os.path.join(path, '*.wav')):
    rate, data = read(filename)
    data = npy.array(data)
    audio = npy.mean(data,1)
    _, time, spectrum = plotSpec(audio, rate)
    
    audio_lengths.append(len(energyFrames(time, spectrum)))

maximum = max(audio_lengths)