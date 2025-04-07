import tensorflow as tf
import numpy as np
from style_transfer_tf import gram_matrix, get_model
from image_loader_tf import load_img, tensor_to_image
import time
import matplotlib.pyplot as plt
import os

# Setup
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

content_path = 'content.jpg'
style_path = 'style.jpg'

content_image = load_img(content_path)
style_image = load_img(style_path)

model, style_layers, content_layers = get_model()
num_style_layers = len(style_layers)
num_content_layers = len(content_layers)

def get_feature_representations(model, content_img, style_img):
    content_outputs = model(content_img)
    style_outputs = model(style_img)

    style_features = [gram_matrix(style_layer) for style_layer in style_outputs[:num_style_layers]]
    content_features = content_outputs[num_style_layers:]
    return style_features, content_features

style_targets, content_targets = get_feature_representations(model, content_image, style_image)

generated_image = tf.Variable(content_image, dtype=tf.float32)
optimizer = tf.optimizers.Adam(learning_rate=0.02)

style_weight = 1e-2
content_weight = 1e4

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        outputs = model(generated_image)
        style_outputs = outputs[:num_style_layers]
        content_outputs = outputs[num_style_layers:]

        style_loss = tf.add_n([tf.reduce_mean((gram_matrix(style_output) - target)**2)
                               for style_output, target in zip(style_outputs, style_targets)])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_output - target)**2)
                                 for content_output, target in zip(content_outputs, content_targets)])
        content_loss *= content_weight / num_content_layers

        loss = style_loss + content_loss

    grad = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

# Train the model
epochs = 20
steps_per_epoch = 100

start = time.time()
for n in range(epochs):
    for m in range(steps_per_epoch):
        train_step(generated_image)
        print(f"Epoch {n+1}, Step {m+1} completed")
end = time.time()
print(f"Total time: {end-start:.1f} seconds")

# Save output image
result = tensor_to_image(generated_image)
result.save("outputs/stylized_tf6.jpg")
print("Stylized image saved to outputs/stylized_tf6.jpg")

# ✅ Save the trained image tensor as a checkpoint
ckpt = tf.train.Checkpoint(image=generated_image)
ckpt.save("checkpoints/stylized_image")
print("Model checkpoint saved in checkpoints/")
