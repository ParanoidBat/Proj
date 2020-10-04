# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:50:07 2019

@author: Batman
"""
import numpy as npy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
#import sounddevice as sd
from scipy.io.wavfile import read
from scipy import signal
import pickle

class Preprocessing:
    
    def plotTimAmp(self, audio, rate, show=False):
        if show:
             plt.plot(npy.arange(audio.shape[0])/rate,audio) # time-amplitude plot
             plt.title(audio)
             plt.xlabel("Time - s")
             plt.ylabel("Amplitude")
             plt.show()
        
    def plotSpec(self, audio, rate, show=False):
        freq, time, sx = signal.spectrogram(audio, fs=rate, window="hamming", nperseg=1024, noverlap=924, detrend=False, scaling="spectrum")
        
        if show:
            plt.pcolormesh(time, freq/1000, 10*npy.log10(sx), cmap="viridis") #log the frequency axis for better readability
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (kHz)")
            plt.title("Spectrogram")
            #plt.ylim(0,10)
            plt.show()
        
        return freq, time, sx
    
    def energyFrames(self, time, spectogram):
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
    
    #segmentation of the data w.r.t peaks
    def segment(self, energy_frames):
        # first segment == 20000. afterwards, find segments relatively; according to found local maxima
        segments = []
        segment_vals = []
        centre_vals = [] # centered b/w ith and kth index, when extending segment
        seg_indices = []
        peaks = [] # indices
        i = k = peak = 0 # main iterator, local iterator, local peak storage
        
        while i < len(energy_frames):
            if energy_frames[i] >=20000.0:
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
                        
                        if energy_frames[k] <= peak*0.6 and peak > 1000: # if next value is less than 2/3rd of value in 'peak', we've found the peak
                            peaks.append(energy_frames.index(peak))
                            k+=1
                            break
                        
                        elif energy_frames[k] == peak or energy_frames[k] > peak: # if next value is greater than value in 'peak', update it (updation is also being done right below' while')
                            peak = energy_frames[k]
                    
                    except IndexError:
                        k-=1
                        break
                        
                # extend the segment to find a transition point
                while k < len(energy_frames):
                    try:
                        # if the value is atleast 2/3rd of the peak
                        # and difference of last segment and peak is atleast 1/8th of the peak
                        if energy_frames[k] <= peak*0.6 and peak > 1000: 
                            
                            # find transition point
                            while k < len(energy_frames):
                                if energy_frames[k] < energy_frames[k+1]: # if transition point is found
    #                                k+=1
                                    raise IndexError # this is used as an escape from main while loop
                                else:
                                    k+=1
                                    
                        else:
                            k+=1
                        
                    except IndexError:
    #                    k-=1 # get last index
                        break
                    
                temp_seg = centre_vals[:] + energy_frames[i:k+1] if centre_vals else energy_frames[i:k+1] # k+1 is value right before plot begins to rise
                
                if centre_vals: del centre_vals[:]
                i = k = k+1
                
                if temp_seg:
                    segments.append(temp_seg[:])
                    seg_indices.append(energy_frames.index(temp_seg[0]))
                    
                    del temp_seg[:]
            
        return segments, seg_indices, peaks
    
    #create the pattern of v,c,s from segmented data
    def createPattern(self, segments, peaks, energy_frames):
        pattern = []
        i= j = 0
        
        # dont pass empty list
        try:
            if not segments[j]:
                pass
        except IndexError:
            j+=1
        
        # if between range, is a consonant, else a vowel
        if max(segments[j]) > 1000 and max(segments[j]) <= 10000:
            pattern.append("c")
            j+=1
            
        elif max(segments[j]) > 10000:
            pattern.append("v")
            j+=1
            i=+1 # next block of code recognizes vowels only, we've recognized one here, so update the iterator
        
        for seg in segments:
            
            seg = segments[j]
            
            if seg:
                try:
                    if max(seg) == energy_frames[peaks[i]]: # if there's a peak in the segment, it's a vowel
                        pattern.append("v")
                        i+=1
                        j+=1
                    
                    else: # we are only looking for vowels, so if not found. continue
                        j+=1
                        continue
                        
                    if len(pattern) > 1:
                        # 2 vowels can't be written together (except for above case). if such a case is found, insert a 'c' b/w them
    
                        if pattern[-2] == "v" and pattern[-1] == "v":
                            pattern.insert(-1, "c")
                
                except IndexError:
                    break
            
        return "".join(map(str, pattern)) # return string, not list
    
    #a generalized plotting function
    def plotStuff(self, data, title, xlabel, ylabel, xlim1= 0, xlim2= 0, ylim1 = 0, ylim2= 0): # a helper function to plot any data
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if xlim1 != 0 and xlim2 != 0:
            plt.xlim(xlim1,xlim2)
            
        if ylim1 != 0 and ylim2 != 0:
            plt.ylim(ylim1, ylim2)
            
        plt.show()
    
    #scale down the energies to give consistant data with the referenced data (eliminating effect of loud sound)
    def scaleDown(self, scale_to, segment, peak):
        factor = scale_to/peak #get the factor to be multiplpied with the segment, to scale them down
        
        for i in range(len(segment)):
            segment[i] = segment[i]*factor
    
    def scaleUp(self, scale_to, segment, peak):
        factor = scale_to/peak
        
        for i in range(len(segment)):
            segment[i] = segment[i]*factor
    
    #smooth energy frames data to eliminate noise
    def smooth(self, data, size, mean): 
        new_data = []
        factor = mean//2
        i = factor
    
        while(i< size-factor):
            new_data.append(sum(data[i-factor:i+factor])/mean) #take mean of 'mean' values, taking i'th value as the pivot
            i+=1
            
        return new_data
    
    
    def zeroCrossingRate(self, wave):
        rate = []
        
        # taking 19ms window. 924 samples
        samples = 924
        
        above = True # boolean to check if value is above or below zero
        prev_state = above # remember the previous state to determine zero crossing
        
        try:
            while 1:
                # don't let go out of bounds
                if samples >= len(wave) or samples > len(wave) - 924:
                    break
                
                counter = 0 # count number of times zero is crossed
                
                # set bool to determine crossing
                if wave[samples - 924] > 0:
                    above = True
                else:
                    above = False
                    
                prev_state = above
                
                for i in range(samples - 924, samples):
                    if wave[i] > 0:
                        above = True
                    else:
                        above = False
                        
                    if prev_state is not above: # the state changed, meaning zero is crossed
                        counter+=1
                        
                    prev_state = above
                    
                samples +=924
                rate.append(counter/0.019)
                    
        except IndexError:
            rate.append(counter/0.019)
            print("Error: Index Is Out Of Bounds!")
            
        return npy.array(rate)
    
    
    def zcrSmooth(self, zcr):
        # staright lines are plotted from peak to peak.
        
        indices = signal.find_peaks(zcr)[0]
    
        new_data = []
        
        for x in indices:
            new_data.append(zcr[x])
        
        return npy.array(new_data)
    
    # get the factor to enable mapping of segments/peaks of energy frames to zcr
    def calcMappingFactor(self, ef_length, zcr_length):
        return ef_length/zcr_length
    
    # map segments/peaks from energy frames to zcr
    def mapEnergyToZcr(self, factor, peaks, segments):
        indices = []
        
        # map peaks
        try:
            if peaks is None: raise IndexError
            
            for i in range(len(peaks)):
                indices.append(peaks[i]//factor)
        except IndexError:
            pass
        
        # map segments
        try:
            if segments is None: raise IndexError
            
            for i in range(len(segments)):
                indices.append(segments[i]//factor)
        except IndexError:
            pass
        
        return npy.array(indices, dtype = npy.int32)
    
    # get the data between consecutive peaks
    def peak2peak(self, peaks, energy_frames):
        data = []
        i = 0
        
        while i < len(peaks) -1:
            data.append(energy_frames[peaks[i]: peaks[i+1] +1])
            i+=1
            
        return data
    
    # pad the data with 0's for scaling purpose to be used in neutral network
    def padding(self, pad_with):
        new_list = [0]*pad_with
        return new_list
    
    def writeToFile(self, _from, _to, vector, prop):
        # max ef is 278
        pad = length = i = 0
        data = []
        
        with open("zcr_peaks.txt", "a") as file:        
            for f in _from:            
                length = len(vector[_from[i] : _to[i]]) # get length of said segment
                
                # if padding length is changed. change in ARIJ app, class: Utility, function: organize
                if(length < 300): pad = 300 - length # if padding is needed
    
                data = self.padding(pad) # add padding
                
                for x in vector[_from[i] : _to[i]]: #appendd data from segment to list after padding
                    data.append(x)
                
                for d in data:
                    file.write(str(d) + ',') # write to file
                
                # reset varaibles
                del data[:] 
                length = 0
                
                file.write(prop[i] + "\n") # append property
                
                i+=1
        
        print("wrote to file")
    
    def recognizeVowels(self, audio_sample, verbose=False, visual=False):
    #    audio_sample = "Samples/Salam12.wav"
        
        sample_rate, wave_data = read(audio_sample)
        data_array = npy.array(wave_data)
        
        audio = npy.mean(data_array, 1) #make it into mono channel
        
#        if verbose: sd.play(audio)
        
        freq, time, sx = self.plotSpec(audio, sample_rate, visual) #get spectrogram
        if verbose: print("freq: ", freq, "\ntime: ", time, "\nspectrum: ", sx)
        
        ef = self.energyFrames(time, sx)
        
        peak = max(ef) # scale the data
        scale_to = 200000
        
        if peak > scale_to:
            self.scaleDown(scale_to, ef, peak)
            if verbose: print("Data scaled up")
        
        elif peak < scale_to:
            self.scaleUp(scale_to, ef, peak)
            if verbose: print("Data scaled down")
        
        self.s_ef = self.smooth(ef, len(ef), 7)
        
        self.segments, self.seg_start, self.peaks = self.segment(self.s_ef) #get segments, their indices and peaks' indices
        
    #    pattern = createPattern(segments, peaks, s_ef)
        
        zero_crossing_rate = self.zeroCrossingRate(audio)
        self.smooth_zcr = self.zcrSmooth(zero_crossing_rate)
        
        if visual:
            ##plotting##
            mapped_values = self.mapEnergyToZcr(self.calcMappingFactor(len(self.s_ef), self.smooth_zcr.size), None, self.seg_start)
            contour = npy.array(self.s_ef)
            indices = npy.array(self.seg_start)
            peakses = npy.array(self.peaks)
            
            #mark segments and peaks on energy contour
            plt.plot(contour)
            plt.plot(indices, contour[indices],'x')
            plt.plot(peakses, contour[peakses], 'v')
            plt.title("Energy contour")
            plt.xlabel("Time - ds (deca sec)")
            plt.ylabel("Energy")
            plt.show()
            
            # wave
            plt.figure(figsize=(6, 9)) # x, y size in inches
            plt.subplots_adjust(hspace = 0.3) # vertical space between subplots in inches
            
            plt.subplot(211) # rows, columns, index
            plt.title("Wave")
            plt.xlabel("Time - s")
            plt.ylabel("Amplitude")
            plt.plot(npy.arange(audio.shape[0])/sample_rate,audio)
            
            # zcr
            plt.subplot(212)
            plt.plot(self.smooth_zcr)
            plt.plot(mapped_values, self.smooth_zcr[mapped_values], 'x')
            plt.title("Zero Crossing Rate")
            plt.xlabel("Time - hs (hecto sec)")
            plt.ylabel("Rate")
            
            plt.show()
            
        return self.peaks, self.seg_start
    
    def getCrests(self):
        return self.peaks
    
    def getTroughs(self):
        return self.seg_start
    
    def getEnergy(self):
        return self.s_ef
    
    def getZCR(self):
        return self.smooth_zcr.tolist()
    
    def getCrestsZCR(self):
        return self.mapEnergyToZcr(self.calcMappingFactor(len(self.s_ef), self.smooth_zcr.size), self.peaks, None).tolist()
    
    def getTroughsZCR(self):
        return self.mapEnergyToZcr(self.calcMappingFactor(len(self.s_ef), self.smooth_zcr.size), None, self.seg_start).tolist()


class Predictor:

    def getPredictions(self, file, testing_samples):
        with open(file, 'rb') as f:
            model = pickle.load(f)
        
        self.prediction = model.predict(testing_samples)
        
        return self.prediction
    
    def getPredictionsARIJ(self, file, data): # data is received as single array
        """
        getPredictions implementation for Android App: ARIJ
        """
        with open(file, 'rb') as f:
            model = pickle.load(f)
        
        # args has lists of regions
        pad = 300
        pairs = len(data)//pad # 300 padded each sample
        
        samples = npy.zeros((pairs, pad))
        
        for i in range(pairs):
            for j in range(pad):
                samples[i, j] = data[(i*pad) +j]
        
        self.prediction = model.predict(samples)
        
        return self.prediction
        
    
    def getPattern(self):
        prop = {"v": 0, "c": 1, "f": 2}
        prop_keys = list(prop.keys())
        pattern = ""
        result = [False]*3
        i = 0
        
        try:
            for predict in self.prediction: # prediction is list of lists. predict is single list in it
                for p in predict: # list has 3 values
                    if p > 0.55:
                        result[i] = True # is vowel|consonant|fricative or multiple
                        i+=1
                
                i = 0
                
                # case for multiple not handled
                for r in result:
                    if r:
                        pattern += prop_keys[i] # append property according to index that has true
                        break
                    i+=1
                
                # reset
                for r in result:
                    r = False
                i = 0
        except:
            pass
#            raise Exception # catch in android as PyException
        
        return pattern


class Training:
    def __init__(self):
        self.indices = {"v": 0, "c": 1, "f": 2} # used to set ouput vector
        self.NUM_INPUTS = 300 # per sample 300 inputs
        self.NUM_OUTPUTS = 3 # v, c, f
        self.NUM_TRAINING_SETS = 785
        self.training_inputs = npy.zeros((self.NUM_TRAINING_SETS, self.NUM_INPUTS))
        self.training_outputs = npy.zeros((self.NUM_TRAINING_SETS, self.NUM_OUTPUTS))
    
    def get_index(self, r): # used to populate input vector
        for i in range(r):
            yield i
    
    def train(self, troughs, crests, save_to):
        files = [troughs, crests]
        
        i = self.get_index(self.NUM_TRAINING_SETS)
        
        for f in files:
            with open (f, "r") as file:
                for line in file:
                    tmp = line.rstrip("\n").split(",")
                    tmp2 = ""
                    for e in tmp[-1:]: tmp2 += e # get the label
                    
                    del tmp[len(tmp) - 1 :]
                    tmp = list(map(float, tmp))
                    
                    # set the input vector
                    try: index = next(i)
                    except: pass
                    
                    for j in range(self.NUM_INPUTS):
                        self.training_inputs[index, j] = tmp[j]
                    
                    # set output vector
                    self.training_outputs[index, self.indices.get(tmp2)] = 1
        
        model = RandomForestRegressor(n_estimators=1000, max_features=3, verbose=1).fit(self.training_inputs, self.training_outputs)

        pickle.dump(model, open(save_to, 'wb'))