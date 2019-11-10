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
     plt.plot(npy.arange(audio.shape[0])/rate,audio) # time-amplitude plot
     plt.title("Audio wave")
     plt.xlabel("Time - s")
     plt.ylabel("Amplitude")
     plt.show()
    
def plotSpec(audio, rate):
    freq, time, sx = signal.spectrogram(audio, fs=rate, window="hamming", nperseg=1024, noverlap=924, detrend=False, scaling="spectrum")
    plt.pcolormesh(time, freq/1000, 10*npy.log10(sx), cmap="viridis") #log the frequency axis for better readability
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title("Spectrogram - audio")
    #plt.ylim(0,10)
    plt.show()
    
    return freq, time, sx

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

def segment(energy_frames):
    # first segment == 25000. afterwards, find segments relatively; according to found local maxima
    segments = []
    segment_vals = []
    centre_vals = [] # centered b/w ith and kth index, when extending segment
    seg_indices = []
    peaks = []
    i = k = peak = 0 # main iterator, local iterator, local peak storage
    
    while i < len(energy_frames):
        if energy_frames[i] >=25000.0:
            k = i
            # the segment should start where the plot begins to rise. So when segment is found, traverse back to find a value
            #greater than previous one, and update the segment to start/stop at that value
            while 1:
                if energy_frames[k] < energy_frames[k-1]:
                    break
                else:
                    k-=1
            
            #store the segments
            del segment_vals[k+1:]
            segments.append(segment_vals[:])
            
            if segment_vals: # if list is not null
                seg_indices.append(energy_frames.index(segment_vals[0])) # get the index of the segment
            
            centre_vals = energy_frames[k:i]  # prepend to next segment
            
            print ("segmented at ", energy_frames[k])
            del segment_vals[:]
            
        else:
            segment_vals.append(energy_frames[i])
            i+=1
            continue
            
        # next segmentations are done through different algorithm
        k = i
        
        while k < len(energy_frames):
            peak = energy_frames[k]
            while 1: # find local maxima/peak, break when found
                try:
                    k+=1
                    
                    if energy_frames[k] <= peak/1.5: # if next value is less than 2/3rd of value in 'peak', we've found the peak
                        peaks.append(energy_frames.index(peak))
                        break
                    
                    elif energy_frames[k] == peak or energy_frames[k] > peak: # if next value is greater than value in 'peak', update it (updation is being done right below' while')
                        peak = energy_frames[k]
                
                except IndexError:
                    k-=1
                    break
                    
            # extend the segment to find a transition point
            while k < len(energy_frames):
                try:
                    if energy_frames[k] <= peak/1.5: # if the value is atleast 2/3rd of the peak
                        
                        # find transition point
                        while k < len(energy_frames):
                            if energy_frames[k] < energy_frames[k+1]: # if transition point is found
                                k+=1
                                raise IndexError # this is used as an escape from main while loop
                            else:
                                k+=1
                                
                    else:
                        k+=1
                    
                except IndexError:
                    k-=1 # get last index
                    break
                
            temp_seg = centre_vals[:] + energy_frames[i:k+1] if centre_vals else energy_frames[i:k+1] # k+1 is value right before plot begins to rise
            
            if centre_vals: del centre_vals[:]
            i = k = k+1
            
            if temp_seg:
                segments.append(temp_seg[:])
                seg_indices.append(energy_frames.index(temp_seg[0]))
                
                del temp_seg[:]
        
    return segments, seg_indices, peaks

def createPattern(segments, peaks, energy_frames):
    pattern = []
    i= 0
    
    for seg in segments:
        if seg:
            try:
                if max(seg) == energy_frames[peaks[i]]: # if there's a peak in the segment, it's a vowel
                    pattern.append("v")
                    i+=1
                
                else: # we are only looking for vowels, so if not found. continue
                    continue
                
                if len(pattern) >= 1: # if segment ends at value 2/3rd of peak, means there's another peak close ahead, thus a vowel. Represent 2 vowels with 'V'
                    if seg[-1:] >= max(seg)/1.7:
                        pattern[-1:] = "V"
                        i+=1 # next peak has been taken care of, move onwards
#                        continue
                    
                if len(pattern) > 1:
                    # 2 vowels can't be written together (except for above case). if such a case is found, insert a 'c' b/w them
                    
                    if (pattern[-2: -1] == "v" or "V" ) and (pattern[-1:] == "v" or "V"): 
                        if not seg[0] < 1000.0: # if segment terminals haven't dived into silence region
                            pattern.insert(-1, "c")
                        else:
                            pattern.insert(-1, "s")
            
            except IndexError:
                break
        
    return "".join(map(str, pattern)) # return string, not list

def plotStuff(data, title, xlabel, ylabel, xlim1= 0, xlim2= 0, ylim1 = 0, ylim2= 0): # a helper function to plot any data
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if xlim1 != 0 and xlim2 != 0:
        plt.xlim(xlim1,xlim2)
        
    if ylim1 != 0 and ylim2 != 0:
        plt.ylim(ylim1, ylim2)
        
    plt.show()
    
def scaleDown(scale_to, segment, peak): #scale down the energies to give consistant data with the referenced data (eliminating effect of loud sound)
    factor = scale_to/peak #get the factor to be multiplpied with the segment, to scale them down
    
    for i in range(len(segment)):
        segment[i] = segment[i]*factor
        
def smooth(data, size): #smooth data to eliminate noise
    new_data = []
    i = 3

    while(i< size-3):
        new_data.append(sum(data[i-3:i+3])/7) #take mean of 7 values, taking i'th value as the pivot
        i+=1
        
    return new_data
    
def recognizeVowels(audio_sample):
#audio_sample = "whatsapp chalao2.wav"
    sample_rate, wave_data = read(audio_sample)
    data_array = npy.array(wave_data)
    audio = npy.mean(data_array,1) #make it into mono channel
    
    freq, time, sx = plotSpec(audio, sample_rate) #get spectrogram
    
    ef = energyFrames(time, sx)
    
    peak = max(ef) #if data's highest value is above threshold, scale it down
    scale_to = 200000
    if peak >scale_to:
        scaleDown(scale_to, ef, peak)
    
    s_ef = smooth(ef, len(ef))
    
    segments, seg_start, peaks = segment(s_ef) #get segments, their indices and peaks' indices
    
    pattern = createPattern(segments, peaks, s_ef)
    
    contour = npy.array(s_ef)
    indices = npy.array(seg_start)
    peakses = npy.array(peaks)
    
    #mark segments and peaks on energy contour
    plt.plot(contour)
    plt.plot(indices, contour[indices],'x')
    plt.plot(peakses, contour[peakses], 'v')
    plt.title("Energy contour - "+ audio_sample)
    plt.xlabel("Index")
    plt.ylabel("Energy")
    #plt.ylim(0,1200)
    #plt.xlim(2000,2500)
    plt.show()
    
    return pattern 