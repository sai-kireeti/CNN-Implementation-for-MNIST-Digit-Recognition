#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, utils
from sklearn.datasets import fetch_openml

# Load the MNIST dataset
mnist = fetch_openml(name='mnist_784', version=1, parser='auto')  # Suppresses FutureWarning

# Extract features and labels
X = np.array(mnist.data)
y = np.array(mnist.target)

# Normalize and reshape data for CNN input
X = X.reshape((X.shape[0], 28, 28, 1)).astype('float32') / 255.0  # Normalize pixel values to range [0, 1]
y = utils.to_categorical(y, 10)  # Assuming there are 10 classes (digits 0-9)

# Build the CNN Model
def create_model():
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(28, 28, 1)),
        # Convolutional layers
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        # Fully connected layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # Output layer with softmax activation for classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Perform K-Fold Cross Validation
def kfold_cross_validation(X, y, n_splits=5, batch_size=128, epochs=10):
    kf = KFold(n_splits=n_splits)
    fold_no = 1
    losses = []
    accuracies = []

    for train, test in kf.split(X):
        print(f'Training fold {fold_no}...')
        model = create_model()  # Create a new model for each fold
        history = model.fit(X[train], y[train], 
                            batch_size=batch_size, epochs=epochs, 
                            validation_data=(X[test], y[test]),
                            verbose=1)  # Enable verbosity during training
        
        scores = model.evaluate(X[test], y[test], verbose=0)
        print(f'Score for fold {fold_no}: Loss = {scores[0]}; Accuracy = {scores[1]*100}%')
        losses.append(scores[0])
        accuracies.append(scores[1])
        fold_no += 1

    # Average scores after cross-validation
    print(f'Average loss: {np.mean(losses)}, Average Accuracy: {np.mean(accuracies)*100}%')

    return losses, accuracies, history

# Plot learning curves
def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Predictions for Confusion Matrix and Classification Report
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Classification Report
    print(classification_report(y_true, y_pred_classes))

# Perform K-Fold Cross Validation
losses, accuracies, history = kfold_cross_validation(X, y)

# Plot learning curves
plot_learning_curves(history)

# Evaluate the model
model = create_model()
model.fit(X, y, epochs=10, batch_size=128, verbose=1)  # Training on entire dataset
evaluate_model(model, X, y)

# Display model summary
model.summary()


# In[ ]:




