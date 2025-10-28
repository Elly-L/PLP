 MNIST Handwritten Digit Classification
# ==========================================

# STEP 1: Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("✅ TensorFlow version:", tf.__version__)

# STEP 2: Load MNIST dataset
# TensorFlow automatically downloads it
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("\n📊 Dataset shapes:")
print("x_train:", x_train.shape, "| y_train:", y_train.shape)
print("x_test:", x_test.shape, "| y_test:", y_test.shape)

# STEP 3: Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension (needed for CNN input)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("✅ Normalized and reshaped data:", x_train.shape)

# STEP 4: Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# STEP 5: Train model
# (increase epochs if accuracy <95%)
history = model.fit(
    x_train, y_train,
    epochs=6,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# STEP 6: Plot training accuracy & loss
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# STEP 7: Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# STEP 8: Visualize predictions on 5 random samples
import random
indices = random.sample(range(len(x_test)), 5)

plt.figure(figsize=(10,3))
for i, idx in enumerate(indices):
    img = x_test[idx]
    true_label = y_test[idx]
    pred_label = np.argmax(model.predict(img[np.newaxis, ...]))
    
    plt.subplot(1,5,i+1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}",
              color='green' if pred_label==true_label else 'red')
    plt.axis('off')
plt.tight_layout()
plt.show()

# STEP 9: Save the model (optional)
model.save("mnist_cnn_model.h5")
print("\n💾 Model saved as mnist_cnn_model.h5")