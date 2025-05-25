import pandas as pd

# Load the CSV file
df = pd.read_csv('/kaggle/input/soilclass/soilclassification/train_labels1.csv')  # Replace with your CSV path
print(df.head())  # Check if data is loaded correctly

from sklearn.model_selection import train_test_split

# Split data into training (80%) and validation (20%)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

# ============ IMPORTS ============
import pandas as pd
import numpy as np
import pickle  # For saving class indices
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============ LOAD DATA ============
train_df = pd.read_csv('/kaggle/input/soilclass/soilclassification/train_labels1.csv')
val_df = train_df.sample(frac=0.2, random_state=42)  # Simple validation split

# ============ CREATE GENERATORS ============
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='/kaggle/input/soilclass/soilclassification/train',
    x_col='image_id',
    y_col='soil_type',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory='/kaggle/input/soilclass/soilclassification/train',
    x_col='image_id',
    y_col='soil_type',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# ============ SAVE CLASS INDICES ============
with open('class_indicesx.pkl', 'wb') as f:
    pickle.dump(train_generator.class_indices, f)
print("Class indices saved. Class mapping:", train_generator.class_indices)

# ============ BUILD MODEL ============
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation (to prevent overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='/kaggle/input/soilclass/soilclassification/train',  # Update path
    x_col='image_id',  # Column with image names
    y_col='soil_type',   # Column with labels
    target_size=(150, 150),  # Resize images
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory='/kaggle/input/soilclass/soilclassification/train',
    x_col='image_id',
    y_col='soil_type',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Get number of classes dynamically
num_classes = len(train_generator.class_indices)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_generator,
    epochs=80,  # Train for 10 rounds (increase if needed)
    validation_data=val_generator
)

import matplotlib.pyplot as plt

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

model.save('soil_classifierrx3final.h5')
print("Model and class indices saved. Download:")


# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

from IPython.display import FileLink
FileLink('soil_classifierrx3ffinal.h5')  # Click the link that appears to download


# ============ IMPORTS ============
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============ LOAD MODEL & CLASS NAMES ============
model = load_model('/kaggle/input/model3id/tensorflow1/default/1/soil_classifierrx3.h5')  # Update path
with open('/kaggle/input/secondtime/tensorflow1/default/1/class_indicesx.pkl', 'rb') as f:
    class_names = list(pickle.load(f).keys())
print("Class Names:", class_names)  # Verify order matches training

# ============ LOAD TEST DATA ============
test_df = pd.read_csv('/kaggle/input/newtest/test_ids1.csv')  # Update path
print("Test Samples:", len(test_df))

# ============ CREATE TEST GENERATOR ============
test_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for testing

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='/kaggle/input/soilclass/soilclassification/test',  # Update path
    x_col='image_id',  # Column with image filenames
    y_col=None,  # No labels
    target_size=(150, 150),  # Must match training size
    batch_size=32,
    class_mode=None,  # Important for prediction
    shuffle=False  # Maintain original order
)

# ============ GENERATE PREDICTIONS ============
predictions = model.predict(test_generator)
predicted_indices = np.argmax(predictions, axis=1)
test_df['soil_type'] = [class_names[i] for i in predicted_indices]

# ============ SAVE RESULTS ============
submission_df = test_df[['image_id', 'soil_type']]  # Keep only required columns
submission_df.to_csv('submission5.csv', index=False)
print("Submission file saved! Preview:")
print(submission_df.head())

# ============ DOWNLOAD ============ 
# Click the 📁 icon in Kaggle's output sidebar to download submission.csv

from IPython.display import FileLink
FileLink('submission4.csv')  # Click the link that appears to download
