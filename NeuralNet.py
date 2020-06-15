# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:39:23 2020

@author: HP
"""
import math
import numpy as npy
import random

def get_index(r): # used to populate input vector
    for i in range(r):
        yield i

indices = {"s": 0, "v": 1, "c": 2, "f": 3} # used to set ouput vector

# nueral net functions

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
NUM_INPUTS = 278 # each sample has 278 elements
NUM_HIDDEN_NODES = 278
NUM_OUTPUTS = 4 # s, v, c, f
LR = 0.1

hidden_layer = [0]*NUM_HIDDEN_NODES
output_layer = [0]*NUM_OUTPUTS

hidden_layer_bias = [0]*NUM_HIDDEN_NODES
output_layer_bias = [0]*NUM_OUTPUTS

hidden_weights = npy.zeros((NUM_INPUTS, NUM_HIDDEN_NODES)) # 2d matrix
output_weights = npy.zeros((NUM_HIDDEN_NODES, NUM_OUTPUTS))

NUM_TRAINING_SETS = 1737

#training_inputs = npy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype = npy.float64)
#training_outputs = npy.array([[0.0], [1.0], [1.0], [0.0]], dtype = npy.float64)

training_inputs = npy.zeros((NUM_TRAINING_SETS, NUM_INPUTS))
training_outputs = npy.zeros((NUM_TRAINING_SETS, NUM_OUTPUTS))

files = ["ef_peaks.txt", "ef_segs.txt", "zcr_peaks.txt", "zcr_segs.txt"]

# populate input/output vectors
i = get_index(NUM_TRAINING_SETS)

for f in files:
    with open (f, "r") as file:
        
        for line in file:
            tmp = line.rstrip("\n").split(",")
            tmp2 = ""
            for e in tmp[-1:]: tmp2+= e
            
            del tmp[len(tmp) - 1 :]
            tmp = list(map(float, tmp))
            
            # set the input vector
            try: index = next(i)
            except: pass
            
            for j in range(278):
                training_inputs[index, j] = tmp[j]
            
            # set output vector
            training_outputs[index, indices.get(tmp2)] = 1

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

training_set_order = [x for x in range(NUM_TRAINING_SETS)]

# training
for n in range(1):
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
        
        print("output:", output_layer, "expected output:", training_outputs[i, :])
        
        
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