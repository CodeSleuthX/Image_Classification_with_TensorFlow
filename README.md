# Image_Classification_with_TensorFlow
The CIFAR-10 dataset comprises 60,000 32x32 color images in 10 different classes, making it an ideal playground for honing your image classification skills.

# Why This Project:
- Gain hands-on experience in image classification, a fundamental task in computer vision.
- Develop proficiency in TensorFlow, one of the most popular deep learning frameworks.
- Enhance your understanding of Convolutional Neural Networks and their applications in image recognition.
- Contribute to the growing field of machine learning by mastering the intricacies of model training, evaluation, and optimization.

# Let's go through the code step by step:

```python
# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import BatchNormalization, Dropout
import matplotlib.pyplot as plt
```

This section imports the required libraries. TensorFlow and Keras are used for building and training the neural network. ImageDataGenerator is used for data augmentation, LearningRateScheduler is used to adjust the learning rate during training, and BatchNormalization and Dropout are regularization techniques. Matplotlib is used for plotting.

```python
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
```

Here, the CIFAR-10 dataset is loaded. It consists of 60,000 32x32 color images in 10 different classes, split into training and testing sets.

```python
# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)
```

Data augmentation is applied using the ImageDataGenerator. It introduces variations in the training images through rotations, width and height shifts, and horizontal flips, enhancing the model's ability to generalize.

```python
# Learning Rate Scheduler
def lr_schedule(epoch):
    return 0.001 * (0.1 ** int(epoch / 10))

lr_callback = LearningRateScheduler(lr_schedule)
```

A learning rate scheduler is defined, which adjusts the learning rate during training. In this example, it reduces the learning rate by a factor of 0.1 every 10 epochs.

```python
# Define the model architecture with Batch Normalization
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10)
])
```

The model architecture is defined using a Sequential model. It includes convolutional layers with Batch Normalization, max-pooling layers, a flatten layer, and fully connected layers. The last layer has 10 neurons, corresponding to the 10 classes in CIFAR-10.

```python
# Dropout for Regularization
model.add(Dropout(0.25))
```

A dropout layer is added after the Dense layer for regularization, which randomly drops 25% of the neurons during training to prevent overfitting.

```python
# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

The model is compiled using the Adam optimizer, sparse categorical cross-entropy loss (suitable for integer-encoded labels), and accuracy as the metric.

```python
# Train the model with Data Augmentation and Learning Rate Scheduler
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[lr_callback]
)
```

The model is trained using the fit method. Data augmentation is applied during training, and the learning rate scheduler is used to adjust the learning rate.

```python
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
```

The trained model is evaluated on the test set, and the test accuracy is printed.

```python
# Save the model
model.save("cifar10_image_classifier.h5")
```

The trained model is saved as an HDF5 file.

```python
# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
```

Finally, the training history is plotted using Matplotlib to visualize the accuracy over epochs for both the training and validation sets. This helps in understanding the model's performance and potential overfitting.

# Workflow 
- creating 
- training 
- evaluating 
- saving an image classification model 
