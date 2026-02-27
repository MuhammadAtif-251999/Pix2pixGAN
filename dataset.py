from PIL import Image
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Define the dimensions for resizing
desired_width = 600
desired_height = 400

# Define the number of images for training and testing
num_train = 300  # Set the number of training images
num_test = 10   # Set the number of testing images

# Load and preprocess low-light images
low_light_image_dir = "/kaggle/input/scie-dataset/SCIE/Input"
low_light_image_files = os.listdir(low_light_image_dir)[:num_train + num_test]  # Limit dataset size
src_images = []

for image_file in low_light_image_files:
    image_path = os.path.join(low_light_image_dir, image_file)
    image = Image.open(image_path)
    image = image.resize((desired_width, desired_height))
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
    src_images.append(image)

# Load and preprocess ground truth (high) images
high_image_dir = "/kaggle/input/scie-dataset/SCIE/Label"
high_image_files = os.listdir(high_image_dir)[:num_train + num_test]  # Limit dataset size
gt_images = []

for image_file in high_image_files:
    image_path = os.path.join(high_image_dir, image_file)
    image = Image.open(image_path)
    image = image.resize((desired_width, desired_height))
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
    gt_images.append(image)

# Convert lists to NumPy arrays
src_images = np.array(src_images)
gt_images = np.array(gt_images)

# Split into training and testing sets
train_src = src_images[:num_train]
train_gt = gt_images[:num_train]
test_src = src_images[num_train:num_train + num_test]
test_gt = gt_images[num_train:num_train + num_test]

print('Training images:', len(train_src), 'Testing images:', len(test_src))
print('Shape of training images:', train_src.shape, 'Shape of testing images:', test_src.shape)

# Create dataset dictionary
dataset = {
    "train": (train_src, train_gt),
    "test": (test_src, test_gt)
}