import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

# For Branhcing
model = keras.Sequential()
model.add( layers.Dense(4, activation="relu", input_shape=(60,)) ) # 46 static stats plus 14 branch-specific stats
model.add(layers.Dense(1, activation="relu"))

optimizer = keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=optimizer, loss= "mean_squared_error")
print(model.summary())
model.save('./models/branch_model_in60_lay2_0')

# For Searching
model = keras.Sequential()
model.add( layers.Dense(4, activation="relu", input_shape=(51,)) ) # 46 static stats plus 5 search-node-specific
model.add(layers.Dense(1, activation="relu"))
optimizer = keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=optimizer, loss= "mean_squared_error")
print(model.summary())
model.save('./models/search_model_in51_lay2_0')
