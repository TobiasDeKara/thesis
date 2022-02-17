import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras import activations

optimizer = keras.optimizers.Adam(learning_rate=0.001) # default learning rate == 0.001 

for action_type in ['branch', 'search']:
	if action_type == 'branch':
		# Note on shape: 61 = 47 static stats plus 14 branch-specific stats
		input_shape = 61
	if action_type == 'search':
		# Note on shape 53 = 47 static stats plus 6 search-node-specific
		input_shape = 53

	for reward_format in ['numeric', 'binary']:
		model = keras.Sequential()
		model.add( layers.Dense(4, activation="relu", input_shape=(input_shape,)) ) 
		model.add(layers.Dense(1, activation="relu"))
		if reward_format == 'binary':
			model.add(layers.Dense(1, activation='sigmoid'))

		model.compile(optimizer=optimizer, loss= "mean_squared_error")

		print(model.summary())

		model.save(f'./models/{action_type}_model_in{input_shape}_lay2_rew_{reward_format}')

