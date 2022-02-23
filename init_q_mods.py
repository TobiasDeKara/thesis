import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras import activations

optimizer = keras.optimizers.Adam(learning_rate=0.001) # default learning rate == 0.001 
drop_out = 'no'

for action_type in ['branch', 'search']:
	if action_type == 'branch':
		# Note on shape: 61 = 47 static stats plus 14 branch-specific stats
		input_shape = 61
	if action_type == 'search':
		# Note on shape 53 = 47 static stats plus 6 search-node-specific
		input_shape = 53

	for n_layer in [3, 4]:
		for reward_format in ['numeric', 'binary']:
			model = keras.Sequential()
			if n_layer > 3:
				model.add( layers.Dense(16, activation="relu", input_shape=(input_shape,)) )
				if drop_out == 'yes':
					model.add(layers.Dropout(rate=0.2))
			if n_layer > 2:
				model.add( layers.Dense(8, activation="relu", input_shape=(input_shape,)) )
				if drop_out == 'yes':
					model.add(layers.Dropout(rate=0.2))
			model.add(layers.Dense(4, activation="relu"))
			model.add(layers.Dense(1, activation="relu"))
			if reward_format == 'binary':
				model.add(layers.Dense(1, activation='sigmoid'))

			model.compile(optimizer=optimizer, loss= "mean_squared_error")

			print(model.summary())

			model_name = \
			f'{action_type}_model_in{input_shape}_lay{n_layer}_drop_out_{drop_out}_rew_{reward_format}'

			model.save(f'./models/{model_name}')
