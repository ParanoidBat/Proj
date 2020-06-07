import numpy as npy
from sklearn.neural_network import MLPRegressor

def get_index(r):
    for i in range(r):
        yield i

indices = {"s": 0, "v": 1, "c": 2, "f": 3}

NUM_INPUTS = NUM_HIDDEN_NODS = 278
NUM_OUTPUTS = 4
NUM_TRAINING_SETS = 1737

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

model = MLPRegressor(hidden_layer_sizes=278, activation='logistic', solver='sgd',
                     learning_rate='adaptive', learning_rate_init=0.1, max_iter=NUM_TRAINING_SETS*10000,
                     verbose=True, nesterovs_momentum=False, n_iter_no_change=10000).fit(
                             training_inputs, training_outputs)