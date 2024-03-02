# Image_Classification_with_TensorFlow
The CIFAR-10 dataset comprises 60,000 32x32 color images in 10 different classes, making it an ideal playground for honing your image classification skills.

# Why This Project:
- Gain hands-on experience in image classification, a fundamental task in computer vision.
- Develop proficiency in TensorFlow, one of the most popular deep learning frameworks.
- Enhance your understanding of Convolutional Neural Networks and their applications in image recognition.
- Contribute to the growing field of machine learning by mastering the intricacies of model training, evaluation, and optimization.

# Let's have a look on implementation of a Convolutional Neural Network (CNN) using TensorFlow and Keras for image classification on the CIFAR-10 dataset.

1. **Import Necessary Libraries:**
   ```python
   import tensorflow as tf
   from tensorflow import keras
   from tensorflow.keras import layers
   ```
   This section imports the required libraries, including TensorFlow and its high-level API Keras, which simplifies the process of building and training neural networks.

2. **Load the CIFAR-10 Dataset:**
   ```python
   (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
   ```
   This line loads the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes. It is split into training and testing sets.

3. **Define the Model Architecture:**
   ```python
   model = keras.Sequential([
       layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
       layers.MaxPooling2D((2, 2)),
       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.MaxPooling2D((2, 2)),
       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.Flatten(),
       layers.Dense(64, activation='relu'),
       layers.Dense(10)
   ])
   ```
   The model architecture is defined using the Sequential API. It consists of three convolutional layers with max-pooling, followed by flattening, and two fully connected layers. The output layer has 10 neurons, corresponding to the 10 classes in CIFAR-10.

4. **Compile the Model:**
   ```python
   model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
   ```
   The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss (suitable for integer-encoded labels), and accuracy as the metric.

5. **Train the Model:**
   ```python
   model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
   ```
   The model is trained on the training set for 10 epochs, and validation data is used to monitor the model's performance during training.

6. **Evaluate the Model:**
   ```python
   test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
   print(f'Test accuracy: {test_acc}')
   ```
   The trained model is evaluated on the test set, and the test accuracy is printed.

7. **Save the Model:**
   ```python
   model.save("cifar10_image_classifier.h5")
   ```
   The trained model is saved as an HDF5 file (`cifar10_image_classifier.h5`) for later use or deployment.

# Workflow 
- creating 
- training 
- evaluating 
- saving an image classification model 
