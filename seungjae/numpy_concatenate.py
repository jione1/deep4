import numpy as np
import time

d1 = np.load('npy/g.npy')
d2 = np.load('npy/m.npy')
d3 = np.load('npy/h.npy')
d4 = np.load('npy/j.npy')

model_data = np.concatenate((d1, d2, d3, d4), axis=0)

np.random.shuffle(model_data)

training_data = model_data[:28000, :]
test_data = model_data[28000:, :]
np.save('model_data/training_data.npy', training_data)
np.save('model_data/test_data.npy', test_data)