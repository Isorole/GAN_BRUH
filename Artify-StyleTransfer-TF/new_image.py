import tensorflow as tf
import numpy as np
from style_transfer_tf import gram_matrix, get_model
from image_loader_tf import load_img, tensor_to_image
import time
import matplotlib.pyplot as plt
import os

# Load new content image
new_content_path = 'another one.jpg'  # your new image
new_content_image = load_img(new_content_path)

# Reuse the style_image and pre-trained model
style_targets, _ = get_feature_representations(model, new_content_image, style_image)

# Create a new generated image initialized from the new content
new_generated_image = tf.Variable(new_content_image, dtype=tf.float32)

# Train with same loop (or fewer epochs for speed)
for n in range(epochs):
    for m in range(steps_per_epoch):
        train_step(new_generated_image)
        print(f"New Image — Epoch {n+1}, Step {m+1}")

# Save the stylized version of the new image
result = tensor_to_image(new_generated_image)
result.save("outputs/stylized_new_content.jpg")
print("New stylized image saved.")
