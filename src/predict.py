import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from model import weighted_binary_crossentropy


loaded_weights = np.load('../results/hate_type/weights_tensor.npy')
weights_tensor = tf.constant(loaded_weights, dtype=tf.float32)


try:
  model_hate_binary = load_model('../models/model_binary_hate.h5')
except Exception as e:
  print(f"\033[91mError loading model 'model_binary_hate.h5': {e}\033[0m")

try:
  model_hate_type = load_model("../models/model_hate_type.h5",
                               custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy(weights_tensor)},
                               compile=False)
except Exception as e:
  print(f"\033[91mError loading model 'model_hate_type.h5': {e}\033[0m")


with open('../results/binary_hate/best_threshold.json', 'r') as f:
    data = json.load(f)

best_threshold_binary_hate = data["threshold"]

