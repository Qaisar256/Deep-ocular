import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import AlexNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess your dataset (ODIR and RFMiD)
# ... (code to load and preprocess data)

# Load the pre-trained Deep-Ocular model
base_model = AlexNet(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Load an example image for visualization
image_path = "path_to_example_image.jpg"  # Replace with an actual image path
image = load_img(image_path, target_size=(224, 224))
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array /= 255.0

# Get the predicted class index
predicted_class = np.argmax(model.predict(image_array))

# Calculate gradients for the predicted class
grad_model = Model(inputs=model.inputs, outputs=[model.get_layer('dense_1').output, model.output])
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(image_array)
    loss = predictions[:, predicted_class]
grads = tape.gradient(loss, conv_outputs)[0]

# Calculate weights for the heatmap
heatmap = np.mean(grads, axis=(0, 1))

# Normalize the heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= heatmap.max()

# Resize heatmap to match the image size
heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

# Apply heatmap on the image
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(cv2.cvtColor(image_array[0], cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)

# Visualize the original image and superimposed heatmap
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.title("Grad-CAM Heatmap")
plt.show()
