# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize image pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Class labels for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10)  # 10 output classes
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.2f}')

# Make predictions on test images
probability_model = tf.keras.Sequential([model, layers.Softmax()])
predictions = probability_model.predict(x_test)

# Function to plot a test image with prediction
def plot_prediction(i):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = x_test[i]
    plt.imshow(img)

    pred_label = np.argmax(predictions[i])
    true_label = int(y_test[i])

    color = 'blue' if pred_label == true_label else 'red'
    confidence = 100 * np.max(predictions[i])
    plt.xlabel(f"Predicted: {class_names[pred_label]} ({confidence:.1f}%)\nTrue: {class_names[true_label]}", color=color)

# Plot first 5 test images with predictions
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plot_prediction(i)
plt.tight_layout()
plt.show()
input("Press Enter to exit...")  # Keeps the terminal open