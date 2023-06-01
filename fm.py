import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Make predictions on test images
predictions = model.predict(test_images)

# Show results for multiple test cases
num_test_cases = 5  # Number of test cases to display

for i in range(num_test_cases):
    # Display the test image
    plt.imshow(test_images[i], cmap='gray')
    plt.axis('off')
    plt.show()

    # Get the predicted label
    predicted_label = np.argmax(predictions[i])

    # Get the actual label
    actual_label = test_labels[i]

    # Display the predicted and actual labels
    print('Predicted label:', predicted_label)
    print('Actual label:', actual_label)
    print()
