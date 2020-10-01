# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 18:24:07 2020

@author: HP
"""
"""
Script to be used by android application: ARIJ
"""

import pickle
import numpy as npy


class Predictor:

    def getPredictions(self, testing_samples):
        with open('model3.sav', 'rb') as file:
            model = pickle.load(file)
        
        self.prediction = model.predict(testing_samples)
        
        return self.prediction
    
    def getPattern(self):
        