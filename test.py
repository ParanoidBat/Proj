import numpy as npy
from sklearn.neural_network import MLPRegressor

def get_index(r): # used to populate input vector
    for i in range(r):
        yield i

indices = {"v": 0, "c": 1, "f": 2} # used to set ouput vector
############################
NUM_INPUTS = 300 # per sample 300 inputs
NUM_HIDDEN_NODES = 300
NUM_OUTPUTS = 3 # v, c, f
LR = 0.1

NUM_TRAINING_SETS = 598

training_inputs = npy.zeros((NUM_TRAINING_SETS, NUM_INPUTS))
training_outputs = npy.zeros((NUM_TRAINING_SETS, NUM_OUTPUTS))
testing_samples = npy.zeros((59, NUM_INPUTS))

files = ["troughs.txt", "crests.txt"]
testing_files = ["test_crests.txt", "test_troughs.txt"]

# populate input/output vectors
i = get_index(NUM_TRAINING_SETS)
k = get_index(59)


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
            
            for j in range(NUM_INPUTS):
                
                training_inputs[index, j] = tmp[j]
            
            # set output vector
            training_outputs[index, indices.get(tmp2)] = 1


model = MLPRegressor(hidden_layer_sizes=NUM_INPUTS, activation='relu', solver='lbfgs',
                     max_iter=10000, verbose=True).fit(training_inputs, training_outputs)

for f in testing_files:
    with open (f, "r") as file:
        
        for line in file:
            tmp = line.rstrip("\n").split(",")
            tmp2 = ""
            for e in tmp[-1:]: tmp2+= e
            
            del tmp[len(tmp) - 1 :]
            tmp = list(map(float, tmp))
            
            # set the input vector
            try: index = next(k)
            except: pass
            
            for j in range(NUM_INPUTS):
                
                testing_samples[index, j] = tmp[j]

prediction = model.predict(testing_samples[:, :])