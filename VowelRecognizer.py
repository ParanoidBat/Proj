# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:50:07 2019

@author: Batman
"""
from scipy.io.wavfile import read
import numpy as npy
import matplotlib.pyplot as plt
#import sounddevice as sd
from scipy import signal

def plotTimAmp(audio, rate):
     plt.plot(npy.arange(audio.shape[0])/rate_bit,audio) # time-amplitude plot
     plt.title("Audio wave")
     plt.xlabel("Time - s")
     plt.ylabel("Amplitude")
     plt.show()
    
def plotSpec(audio, rate):
    freq, time, sx = signal.spectrogram(audio, fs=rate_about, window="hamming", nperseg=1024, noverlap=924, detrend=False, scaling="spectrum")
    plt.pcolormesh(time, freq/1000, 10*npy.log10(sx), cmap="viridis") #log the frequency axis for better readability
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title("Spectrogram - audio")
    #plt.ylim(0,10)
    plt.show()
    
    return freq, time, sx

def energyFrames(time, spectogram):
    average =[]
    for i in range(time.size):
        if i+4 >= time.size: break
        #take mean of matrix along x-axis
        vector = npy.mean(sx[:170,i:i+4],1) #frequencies after index 170 are useless. As 8k Hz is the last formant of use
        i+=5
        #get a single value out of the vector, which represents a single value energy magnitude of the frame
        vector_avg = npy.mean(vector)
        average.append(vector_avg)
    
    return average

def segment():
    #silent <1000, segment == 25000
    segments = []
    segment_vals = []
    centre_vals = [] #centered b/w ith and kth index, when extending segment
    seg_start = []
    i = k = 0
    
    while i < len(average):
        if average[i] >=25000.0:
            k = i
            #the segment should start where the plot begins to rise again. So when segment is found, traverse back to find a value
            #greater than previous one, and update the segment to start/stop at that value
            while 1:
                if average[k] < average[k-1]:
                    break
                else:
                    k-=1
            
            #store the segments
            del segment_vals[k+1:]
            segments.append(segment_vals[:])
            seg_start.append(segment_vals[0])
            centre_vals = average[k:i]  #prepend to next segment
            
            print ("segmented at ", average[k])
            
            if max(segment_vals) > 1000.0:
                print("non-silence")
            else:
                print("silence")
            
            del segment_vals[:]
            
            for j in range(i, len(average)):
                if average[j] < 25000.0:
                    k = j
                    
                    while 1:
                        if k+1 == len(average): break #dont wanna go out of bounds
                    
                        if average[k] < average[k+1]:
                            break
                        else:
                            k+=1
                    
                    #while loop breaks where the transition is found, create a new list with both segments' values
                    u_seg_vals = centre_vals[:] + segment_vals[:] + average[j:k+1]
                    
                    segments.append(u_seg_vals[:])
                    seg_start.append(u_seg_vals[0])
                    
                    print ("segmented at ", average[k])
                    print ("non-silence") #this condition can only be met if iterator has passed a value >25000, when going back,
                    #it's obvious the previous region was non-silence
                    
                    del segment_vals[:]
                    del u_seg_vals[:]
                    del centre_vals[:]
                    
                    i = k
                    break
                
                segment_vals.append(average[j])
                
        else: segment_vals.append(average[i])
        i+=1
        
    return segments, seg_start

def createPattern():
    pattern = ""
    
    for seg in segments:
        if max(seg) > 1000 and max(seg) < 300000:
            pattern+="c"
        elif max(seg) >=300000: pattern +="v"
        
    return pattern

def plotStuff(data, title, xlabel, ylabel, xlim1= 0, xlim2= 0, ylim1 = 0, ylim2= 0):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if xlim1 != 0 and xlim2 != 0:
        plt.xlim(xlim1,xlim2)
        
    if ylim1 != 0 and ylim2 != 0:
        plt.ylim(ylim1, ylim2)
        
    plt.show()


#read audio files
rate_names, data_names = read("names.wav")
rate_bit, data_bit = read("bit.wav")
rate_bat, data_bat = read("bat.wav")
rate_about, data_about = read("about.wav")

#store audio samples in numpy array
names_array = npy.array(data_names)
bit_array = npy.array(data_bit)
bat_array = npy.array(data_bat)
about_array = npy.array(data_about)

audio = npy.mean(bit_array[20000:50000],1) #convert to mono channel and trim audio

# plot with time, instead of samples at x-axis
#sd.play(audio)
#length = audio.shape[0]/rate_bat #length of audio (samples/sample rate)
plotTimAmp(audio, rate_bit)

# plot spectogram
freq, time, sx = plotSpec(audio, rate_bit)

#compute high density energy frames, take 5 element window
average = energyFrames(time, sx)
plotStuff(average, "Energy contour", "index", "energy")

#recognize and get silence and non-silence regions
segments, seg_start = segment()

#get the pattern string
pattern = createPattern()

#plot segments over contour
x = npy.array(average)
seg_indices = [average.index(i) for i in seg_start]
y = npy.array(seg_indices)

plt.plot(x)
plt.plot(y,x[y],'x')
plt.show()