import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# part-1(Load MNIST dataset)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Show shape
print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

# Show first image
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

#part-2(Preprocess Data + Build CNN)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Normalize the input data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape input to add channel (grayscale = 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 output classes: digits 0–9
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

#part-3(Evaluate + Predict on Test Image)
import numpy as np

# Pick a random test image
index = np.random.randint(0, len(x_test))
test_image = x_test[index]

# Predict class
prediction = model.predict(np.expand_dims(test_image, axis=0))
predicted_class = np.argmax(prediction)

# Show image
plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_class}")
plt.axis('off')
plt.show()

