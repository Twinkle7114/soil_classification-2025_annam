# ============ IMPORT REQUIRED LIBRARIES ============
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython.display import FileLink

# ============ LOAD LABELS DATA ============
# Load the CSV file containing image IDs and corresponding soil type labels
df = pd.read_csv('/kaggle/input/soilclass/soilclassification/train_labels1.csv')
print(df.head())  # Display the first few rows to confirm the data is loaded properly

# ============ TRAIN-VALIDATION SPLIT ============
# Split the dataset into 80% training and 20% validation data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

# ============ IMAGE DATA AUGMENTATION & GENERATORS ============
# Data augmentation for training set to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values
    rotation_range=20,           # Randomly rotate images
    zoom_range=0.2,              # Randomly zoom
    horizontal_flip=True         # Randomly flip images horizontally
)

# Only rescaling for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

# Create training data generator from dataframe
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='/kaggle/input/soilclass/soilclassification/train',
    x_col='image_id',         # Column containing image filenames
    y_col='soil_type',        # Column containing class labels
    target_size=(150, 150),   # Resize all images to 150x150
    batch_size=32,
    class_mode='categorical'  # Use one-hot encoding for multi-class labels
)

# Create validation data generator
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory='/kaggle/input/soilclass/soilclassification/train',
    x_col='image_id',
    y_col='soil_type',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# ============ SAVE CLASS LABEL MAPPING ============
# Save the class index mapping to use later during inference
with open('class_indicesx.pkl', 'wb') as f:
    pickle.dump(train_generator.class_indices, f)
print("Class indices saved. Class mapping:", train_generator.class_indices)

# ============ BUILD THE CNN MODEL ============
num_classes = len(train_generator.class_indices)  # Get total number of classes dynamically

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # First conv layer
    MaxPooling2D(2, 2),                                                # Downsampling
    Conv2D(64, (3, 3), activation='relu'),                             # Second conv layer
    MaxPooling2D(2, 2),
    Flatten(),                                                         # Flatten to feed into Dense layer
    Dense(128, activation='relu'),                                     # Fully connected layer
    Dense(num_classes, activation='softmax')                           # Output layer with softmax for multi-class
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model architecture
model.summary()

# ============ TRAIN THE MODEL ============
# Fit the model on the training data and validate on validation data
history = model.fit(
    train_generator,
    epochs=80,
    validation_data=val_generator
)

# ============ VISUALIZE TRAINING HISTORY ============
# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ============ SAVE TRAINED MODEL ============
model.save('soil_classifierrx3final.h5')
print("Model saved as 'soil_classifierrx3final.h5'.")

# Evaluate model performance on validation set
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Generate a downloadable link (Kaggle-specific)
FileLink('soil_classifierrx3final.h5')

# ============ LOAD MODEL FOR TESTING ============
# Load the trained model
model = load_model('/kaggle/input/model3id/tensorflow1/default/1/soil_classifierrx3.h5')

# Load class indices used during training
with open('/kaggle/input/secondtime/tensorflow1/default/1/class_indicesx.pkl', 'rb') as f:
    class_names = list(pickle.load(f).keys())
print("Class Names:", class_names)

# ============ LOAD TEST DATA ============
test_df = pd.read_csv('/kaggle/input/newtest/test_ids1.csv')  # CSV with image IDs for testing
print("Test Samples:", len(test_df))

# ============ CREATE TEST DATA GENERATOR ============
# Only rescaling here, no augmentation for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Create test generator for prediction
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='/kaggle/input/soilclass/soilclassification/test',
    x_col='image_id',
    y_col=None,  # No labels in test set
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # Important for predictions
    shuffle=False     # Maintain order of predictions
)

# ============ MAKE PREDICTIONS ============
# Predict probabilities for each class
predictions = model.predict(test_generator)

# Get the class with highest probability for each image
predicted_indices = np.argmax(predictions, axis=1)

# Map indices back to class names
test_df['soil_type'] = [class_names[i] for i in predicted_indices]

# ============ SAVE PREDICTIONS TO CSV ============
submission_df = test_df[['image_id', 'soil_type']]
submission_df.to_csv('submission5.csv', index=False)
print("Submission file saved. Preview:")
print(submission_df.head())

# Download link for the submission file
FileLink('submission5.csv')
