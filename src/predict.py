import numpy as np
import pandas as pd
import tensorflow as tf
import json

from tensorflow.keras.models import load_model
from model import weighted_binary_crossentropy
from data_utils import  preprocess_text


# LOAD THE TENSOR WEIGHTS FOR THE 'model_hate_type'
loaded_weights = np.load('../results/hate_type/weights_tensor.npy')
weights_tensor = tf.constant(loaded_weights, dtype=tf.float32)


# Load the first model
try:
  model_hate_binary = load_model('../models/model_binary_hate.h5')
  print(f"\033[92mModel 'model_binary_hate.h5' loaded successfully\033[0m")
except Exception as e:
  print(f"\033[91mError loading model 'model_binary_hate.h5': {e}\033[0m")

# Load the second model
try:
  model_hate_type = load_model("../models/model_hate_type.h5",
                               custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy(weights_tensor)},
                               compile=False)
  print(f"\033[92mModel 'model_hate_type.h5' loaded successfully\033[0m")
except Exception as e:
  print(f"\033[91mError loading model 'model_hate_type.h5': {e}\033[0m")

# Load the optimal threshold for the first model
with open('../results/binary_hate/best_threshold.json', 'r') as f:
  best_threshold_binary_hate = json.load(f)["threshold"]


df = pd.read_csv('../data/test_comments.csv')
df = preprocess_text(df)