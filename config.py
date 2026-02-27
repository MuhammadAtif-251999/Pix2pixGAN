import os

# Paths
LOW_LIGHT_DIR = "/kaggle/input/scie-dataset/SCIE/Input"
HIGH_LIGHT_DIR = "/kaggle/input/scie-dataset/SCIE/Label"

# Image Dimensions
WIDTH = 600
HEIGHT = 400
CHANNELS = 3

# Training Hyperparameters
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 1.0  # For Adadelta
NUM_TRAIN = 300
NUM_TEST = 10