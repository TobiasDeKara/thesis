import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

# For Branhcing
model = keras.Sequential()
model.add( layers.Dense(4, activation="relu", input_shape=(58,)) ) # 45 static stats plus 13 branch-specific stats
model.add(layers.Dense(1, activation="relu"))
model.compile("adam", "mean_squared_error", metrics=["accuracy"])
# model.fit(np.ones((1,58), dtype=float), np.array([1], dtype=float))
print(model.summary())
model.save('./models/branch_model_in58_lay2_0')

# For Searching
model = keras.Sequential()
model.add( layers.Dense(4, activation="relu", input_shape=(49,)) ) # 45 static stats plus 4
model.add(layers.Dense(1, activation="relu"))
model.compile("adam", "mean_squared_error", metrics=["accuracy"])
# model.fit(np.ones((1,45), dtype=float), np.array([1], dtype=float))
print(model.summary())
model.save('./models/search_model_in49_lay2_0')
