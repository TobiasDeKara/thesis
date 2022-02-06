import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# For Branhcing
branch_model = keras.Sequential()
branch_model.add( layers.Dense(4, activation="relu", input_shape=(61,)) ) # 47 static stats plus 14 branch-specific stats
branch_model.add(layers.Dense(1, activation="relu"))
optimizer = keras.optimizers.Adam(learning_rate=0.0001) # default learning rate == 0.001, 0.00001 was too small
branch_model.compile(optimizer=optimizer, loss= "mean_squared_error")
print(branch_model.summary())
branch_model.save('./models/branch_model_in61_lay2')

# For Searching
#search_model = keras.Sequential()
#search_model.add( layers.Dense(4, activation="relu", input_shape=(53,)) ) # 47 static stats plus 6 search-node-specific
#search_model.add(layers.Dense(1, activation="relu"))
#optimizer = keras.optimizers.Adam(learning_rate=0.001) # default learning rate == 0.001
#search_model.compile(optimizer=optimizer, loss= "mean_squared_error")
#print(search_model.summary())
#search_model.save('./models/search_model_in53_lay2')

