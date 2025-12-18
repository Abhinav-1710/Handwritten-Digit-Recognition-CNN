import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # For better confusion matrix visualization

# --- 1. Data Loading ---
print("--- 1. Loading and Preprocessing Data ---")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# --- 2. Data Preprocessing ---
# Reshape data to include channel dimension (28x28x1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# Normalization: Scale pixel values from 0-255 to 0-1
x_train /= 255
x_test /= 255

# One-Hot Encoding of target labels
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Training Data Shape: {x_train.shape}")
print(f"Test Data Shape: {x_test.shape}")


# --- 3. Model Architecture (Building the CNN) ---
print("--- 2. Building CNN Model ---")

model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Regularization
    Dropout(0.25),
    
    # Flattening and Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # More dropout before final layer
    
    # Output Layer (10 classes with softmax for probability distribution)
    Dense(num_classes, activation='softmax') 
])

# --- 4. Compile the Model ---
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- 5. Train the Model ---
print("--- 3. Training the Model (This will take a few minutes) ---")
epochs = 10 # 10 epochs is sufficient for high accuracy
batch_size = 128

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, # Show progress
                    validation_data=(x_test, y_test))


# --- 6. Save the Trained Model ---
model_save_path = 'model_files/cnn_mnist_model.h5'
model.save(model_save_path)
print(f"\nModel successfully saved to: {model_save_path}")


# --- 7. Evaluation and Analysis ---
print("\n--- 4. Final Evaluation on Test Data ---")
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Test Loss: {loss:.4f}")
print(f"Final Test Accuracy: {acc * 100:.2f}%")


# Plotting Training History
print("\n--- 5. Plotting Training History ---")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model_files/accuracy_loss_plot.png')
plt.show()

# Generate Classification Report and Confusion Matrix
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1) # Convert probabilities to class prediction (0-9)
y_true = np.argmax(y_test, axis=1) # Convert one-hot to class labels

print("\n--- 6. Classification Report ---")
print(classification_report(y_true, y_pred))

# Confusion Matrix Visualization 
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('model_files/confusion_matrix.png')
plt.show()

print("\n--- Project Finished! Check the 'model_files' folder for results. ---")