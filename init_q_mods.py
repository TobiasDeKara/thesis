import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.losses import mean_squared_error
# from tensorflow.keras import activations
action_type = 'branch' # for backward compatibility
lean = True
for model_scope in ['range']: #, 'specific']:
	for reward_format in ['binary']: # , 'big_binary', 'numeric', 'sig_num']:
		for learning_rate in [0.00001]: #, 0.000001]:
			optimizer = keras.optimizers.Adam(learning_rate=learning_rate) # default learning rate == 0.001 
		
			for drop_out in ['yes', 'no']:
				if lean:
					input_shape = 18
				else: 
					# Note on shape: 62 = 47 static stats plus 15 branch-specific stats
					input_shape = 62
		
				for n_layer in [3,4,5,6]:  # [7,8,9]:
					for regularization in [True]: # [True, False]:
						model = keras.Sequential()
		
						if n_layer > 8:
							model.add( layers.Dense(40, activation='relu', input_shape=(input_shape,)) )
							if regularization == True:
								model.add(layers.BatchNormalization())
		
							if drop_out == 'yes':
								model.add(layers.Dropout(rate=0.2))
		
						if n_layer > 7:
							model.add( layers.Dense(32, activation='relu', input_shape=(input_shape,)) )
							if regularization == True:
								model.add(layers.BatchNormalization())
		
							if drop_out == 'yes':
								model.add(layers.Dropout(rate=0.2))
		
		
						if n_layer > 6:
							model.add( layers.Dense(32, activation='relu', input_shape=(input_shape,)) )
							if regularization == True:
								model.add(layers.BatchNormalization())
		
							if drop_out == 'yes':
								model.add(layers.Dropout(rate=0.2))
		
						if n_layer > 5:
							model.add( layers.Dense(16, activation='relu', input_shape=(input_shape,)) )
							if regularization == True:
								model.add(layers.BatchNormalization())
		
							if drop_out == 'yes':
								model.add(layers.Dropout(rate=0.2))
		
						if n_layer > 4:
							model.add( layers.Dense(16, activation='relu', input_shape=(input_shape,)) )
							if regularization == True:
								model.add(layers.BatchNormalization())
		
							if drop_out == 'yes':
								model.add(layers.Dropout(rate=0.2))
		
						if n_layer > 3:
							model.add( layers.Dense(16, activation="relu", input_shape=(input_shape,)) )
							if drop_out == 'yes':
								model.add(layers.Dropout(rate=0.2))
						if n_layer > 2:
							model.add( layers.Dense(8, activation="relu", input_shape=(input_shape,)) )
							if regularization == True:
								model.add(layers.BatchNormalization())
							if drop_out == 'yes':
								model.add(layers.Dropout(rate=0.2))
						model.add(layers.Dense(4, activation="relu"))
						model.add(layers.Dense(1, activation="relu"))
						if reward_format=='binary' or reward_format=='big_binary' or reward_format == 'sig_num':
							model.add(layers.Dense(1, activation='sigmoid'))
		
						model.compile(optimizer=optimizer, loss=mean_squared_error)
		
						print(model.summary())
		
						if lean:
							model_name = f'lean_in{input_shape}_lay{n_layer}_drop_out_{drop_out}_rew_{reward_format}_reg_{regularization}_rate_{learning_rate}_{model_scope}'
						else:
							model_name = f'{action_type}_model_in{input_shape}_lay{n_layer}_drop_out_{drop_out}_rew_{reward_format}_reg_{regularization}_rate_{learning_rate}_{model_scope}'
						# model_name = f'mp_{reward_format}_{model_scope}'
						model.save(f'./models/{model_name}')
