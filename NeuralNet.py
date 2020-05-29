# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:39:23 2020

@author: HP
"""
import math
import numpy as npy
import random
import decimal

decimal.getcontext().prec = 100 # to handle very large numbers

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def init_weight():
    return random.uniform(0, 1)

def shuffle(arr, n):
    if n > 1:
        
        for i in range(n - 1):
            j = int( i + random.randrange(2147483647) / (2147483647 / (n - i) + 1) )
            
            arr[j], arr[i] = arr[i], arr[j]
            
    return None

############################
NUM_INPUTS = 2
NUM_HIDDEN_NODES = 2
NUM_OUTPUTS = 1
LR = 0.1

hidden_layer = [0]*NUM_HIDDEN_NODES
output_layer = [0]*NUM_OUTPUTS

hidden_layer_bias = [0]*NUM_HIDDEN_NODES
output_layer_bias = [0]*NUM_OUTPUTS

hidden_weights = npy.zeros((2,2)) # 2d matrix
output_weights = npy.zeros((2,1))

NUM_TRAINING_SETS = 4

training_inputs = npy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype = npy.float64)
training_outputs = npy.array([[0.0], [1.0], [1.0], [0.0]], dtype = npy.float64)

# initialize hidden weigths
for i in range(NUM_INPUTS):
    for j in range(NUM_HIDDEN_NODES):
        hidden_weights[i,j] = init_weight()

# initialize output weights
for i in range(NUM_HIDDEN_NODES):
    hidden_layer_bias[i] = init_weight()
    
    for j in range(NUM_OUTPUTS):
        output_weights[i,j] = init_weight()

# initialize output layer bias
for i in range(NUM_OUTPUTS):
    output_layer_bias[i] = init_weight()

training_set_order = [0, 1, 2, 3]

# training
for n in range(10000):
    shuffle(training_set_order, NUM_TRAINING_SETS)
    
    for x in range(NUM_TRAINING_SETS):
        i = training_set_order[x]
        
        # forward pass
        for j in range(NUM_HIDDEN_NODES):
            activation = hidden_layer_bias[j]
            
            for k in range(NUM_INPUTS):
                activation+= training_inputs[i,k] * hidden_weights[k,j]
            
            hidden_layer[j] = sigmoid(activation)
            
        for j in range(NUM_OUTPUTS):
            activation = output_layer_bias[j]
            
            for k in range(NUM_HIDDEN_NODES):
                activation += hidden_layer[k] * output_weights[k,j]
            
            output_layer[j] = sigmoid(activation)
        
        print("input:", training_inputs[i, 0], training_inputs[i,1], "output:", output_layer[0], "expected output:", training_outputs[i,0])
        
        
        # back propogation
        delta_output = [0]*NUM_OUTPUTS
        
        for j in range(NUM_OUTPUTS):
            error_output = training_outputs[i,j] - output_layer[j]
            
            delta_output[j] = error_output * d_sigmoid(output_layer[j])
        
        delta_hidden = [0]*NUM_HIDDEN_NODES
        
        for j in range(NUM_HIDDEN_NODES):
            error_hidden = 0.0
            
            for k in range(NUM_OUTPUTS):
                error_hidden+= delta_output[k] * output_weights[j,k]
            
            delta_hidden[j] = error_hidden * d_sigmoid(hidden_layer[j])
        
        for j in range(NUM_OUTPUTS):
            output_layer_bias[j] += delta_output[j] * LR
            
            for k in range(NUM_HIDDEN_NODES):
                output_weights[k,j] += hidden_layer[k] * delta_output[j] * LR
        
        for j in range(NUM_HIDDEN_NODES):
            hidden_layer_bias[j] += delta_hidden[j] * LR
            
            for k in range(NUM_INPUTS):
                hidden_weights[k,j] += training_inputs[i,k] * delta_hidden[j] * LR

# print weights
print("final hidden weights:\n[" )

for j in range(NUM_HIDDEN_NODES):
    print("[")
    
    for k in range(NUM_INPUTS):
        print(hidden_weights[k,j], " ")
    
    print("]")

print("]")

print("final hidden biases:\n[")

for j in range(NUM_HIDDEN_NODES):
    print(hidden_layer_bias[j], " ")

print("]\n")

print("final output weights:")

for j in range(NUM_OUTPUTS):
    print("[")
    
    for k in range(NUM_HIDDEN_NODES):
        print(output_weights[k,j], " ")
    
    print("]\n")

print("final output biases:\n[")

for j in range(NUM_OUTPUTS):
    print(output_layer_bias[j], " ")

print("]\n")