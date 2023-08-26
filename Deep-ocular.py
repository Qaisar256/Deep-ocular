import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import AlexNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess your dataset (ODIR and RFMiD)
# ... (code to load and preprocess data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1.0/255.0
)
datagen.fit(X_train)

# Improved AlexNet architecture with attention and dense layers
base_model = AlexNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Additional Convolutional Layer
x = Conv2D(128, (5, 5), activation='relu', padding='same')(base_model.output)

# Attention Layer
x = Attention()([x, x])

# Additional Convolutional Layer
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

# Global Average Pooling
x = GlobalAveragePooling2D()(x)

# Additional Dense Layers
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

# Output Layer
predictions = Dense(num_classes, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with fine-tuning
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Unfreeze some top layers for fine-tuning
for layer in model.layers[-6:]:
    layer.trainable = True

# Recompile the model after unfreezing
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Implement early stopping to prevent overfitting
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

# Continue training with fine-tuning
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Feature selection using Relief and Recognize by Gradient Boosting
# ... (rest of your code for feature selection and XGBoost classifier)
# Feature selection using ReliefF
relieff = ReliefF()
X_train_selected = relieff.fit_transform(X_train, y_train)
X_test_selected = relieff.transform(X_test)

# Train a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train_selected, y_train)

# Predict the class labels
y_pred = gb_classifier.predict(X_test_selected)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Class mapping for reference
class_mapping = {0: "Glaucoma (GA)", 1: "Diabetic Retinopathy (DR)", 2: "Cataract (CT)", 3: "Normal (NL)"}

# Example prediction for a new retinograph
new_retinograph = ...  # Load and preprocess a new retinograph
new_retinograph_selected = relieff.transform(new_retinograph.reshape(1, -1))
predicted_class_index = gb_classifier.predict(new_retinograph_selected)[0]
predicted_class = class_mapping[predicted_class_index]
print(f"Predicted Class: {predicted_class}")