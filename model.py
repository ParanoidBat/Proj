# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 18:24:07 2020

@author: HP
"""
"""
Script to be used by android application: ARIJ
"""

import pickle

def getPredictions(testing_samples):
    with open('model3.sav', 'rb') as file:
        model = pickle.load(file)
    
    prediction = model.predict(testing_samples)
    
    return prediction