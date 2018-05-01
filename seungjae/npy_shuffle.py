import numpy as np

whole_data = np.load('result.npy')

np.random.shuffle(whole_data)

training_data = whole_data[:10000, :]
test_data = whole_data[10000:, :]

np.save('training_data.npy', training_data)
np.save('test_data.npy', test_data)