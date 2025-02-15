# In the name of God

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

##################### pre-process section #########################

# Load the dataset
# here, as dataset does not have 
file_path = '/home/farhad/Desktop/priv/codes/ml/mlp-iris-classifier/iris_labeled.csv'
data = pd.read_csv(file_path)

# Separate features (X) and labels (y) with selecting entire column by their header
X = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']] # all features columns
y = data[['Species']] # only 1 label column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)  # Initialize the OneHotEncoder with creating an object for it

# Fit the encoder on the training labels only
y_train_encoded = encoder.fit_transform(y_train)

# Transform the test labels using the fitted encoder
y_test_encoded = encoder.transform(y_test)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #  calculates the mean and standard deviation of each feature in X_train, then scales the values
X_test = scaler.transform(X_test) # Uses the same scaling from X_train on X_test without recalculating the mean and standard deviation

# [optional] check pre-process step
print(y_train_encoded.shape)
print(y_train_encoded)

##################### Build the neural network with Sequential model section #########################

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
#     tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer
#     tf.keras.layers.Dense(y_train_encoded.shape[1], activation='softmax')  # Output layer
# ])

##################### Build the neural network with Functional model section #########################

inputs = tf.keras.layers.Input(shape=(X_train.shape[1],))
dense1 = tf.keras.layers.Dense(64, activation='relu')(inputs)
dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
outputs = tf.keras.layers.Dense(y_train_encoded.shape[1], activation='softmax')(dense2)

model = tf.keras.Model(inputs= inputs , outputs=outputs)

####################### compiling the model Section #########################

model.compile(
    optimizer='adam',       # or 'SGD'
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

######### Train the model (with spliting validation data at same time) #######

history = model.fit(
    X_train, y_train_encoded,
    epochs=50,
    batch_size=16,         # default batch_size in keras is 32
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1
)

################# optional###################3
# print history elements
print(history.history)

# note: content of history is an dictionary
# there are 4 keys in this dictionary: accuracy, loss, val_accuracy, val_loss
# each key contains by the number of epochs values

# store values of each key in a variable
accuracy = (history.history['accuracy'])
print(accuracy)

loss = (history.history['loss']) 
print(loss)

val_accuracy = (history.history['val_accuracy'])
print(val_accuracy)

val_loss = (history.history['val_loss'])
print(val_loss)

########################### evaluating Section ###########################

# find best accurary and loss during fitting process
best_train_acc = round(max(history.history['accuracy']), 2)
best_train_loss = f'{min(history.history['loss']):.2f}'
best_val_acc = f'{max(history.history['val_accuracy']):.2f}'
best_val_loss = f'{min(history.history['val_loss']):.2f}'

# Evaluate the model on the test data (final evaluation)
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=1)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Make predictions on the test data
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_encoded, axis=1)
print(predictions)
print(predicted_classes)



# Print classification results
print(f'best Train Accuracy is: {best_train_acc}')
print(f'best Train Loss is: {best_train_loss}')
print(f'best Validate Accuracy is: {best_val_acc}')
print(f'best Validate Loss is: {best_val_loss}')
print("Predicted Classes:", predicted_classes)
print("True      Classes:", true_classes)


########################### Ploting Section ###########################

# plot training accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Accuracy and loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/loss value')
plt.text(0.5, 0.50, f"Best train Accuracy: {best_train_acc}", fontsize=12, color='blue', transform=plt.gca().transAxes)
plt.text(0.5, 0.43, f"Best train loss: {best_train_loss}", fontsize=12, color='red', transform=plt.gca().transAxes)
plt.legend()
plt.tight_layout()
plt.savefig('training accuracy and loss.png', dpi=300) # optional
plt.show()

# plot validation accuracy and loss
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Validation Accuracy and loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/loss value')
plt.figtext(0.5, 0.01, f'best validation accuracy and loss are: {best_val_acc}, {best_val_loss}', fontsize=10, ha='center', color='red')
plt.legend()
plt.tight_layout()
plt.savefig('validation accuracy and loss.png', dpi=300) # optional
plt.show()


# plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.text(0.5, 0.50, f"Best train Accuracy: {best_train_acc}", fontsize=12, color='blue', transform=plt.gca().transAxes)
plt.text(0.5, 0.43, f"Best validation Accuracy: {best_val_acc}", fontsize=12, color='red', transform=plt.gca().transAxes)
plt.legend()
plt.tight_layout()
plt.savefig('trainin and validation accuracy.png', dpi=300) # optional
plt.show()

# plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.text(0.5, 0.50, f"Best train loss: {best_train_loss}", fontsize=12, color='blue', transform=plt.gca().transAxes)
plt.text(0.5, 0.43, f"Best validation loss: {best_val_loss}", fontsize=12, color='red', transform=plt.gca().transAxes)
plt.legend()
plt.tight_layout()
plt.savefig('trainin and validation loss.png', dpi=300) # optional
plt.show()

# plot training and validation loss and accuray (everythings in single plot)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and validation Accuracy and Loss (everythings)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy / Loss')
plt.text(0.5, 0.60, f"Best train Accuracy: {best_train_acc}", fontsize=12, color='blue', transform=plt.gca().transAxes)
plt.text(0.5, 0.52, f"Best validate Accuracy: {best_val_acc}", fontsize=12, color='red', transform=plt.gca().transAxes)
plt.text(0.5, 0.44, f"Best train loss: {best_train_loss}", fontsize=12, color='skyblue', transform=plt.gca().transAxes)
plt.text(0.5, 0.36, f"Best validation loss: {best_val_loss}", fontsize=12, color='orange', transform=plt.gca().transAxes)
plt.legend()
plt.tight_layout()
plt.savefig('trainin-validation accuracy-loss.png', dpi=300) # optional
plt.show()

# plot training and validation accuracy /  loss side-by-side

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.text(0.5, 0.50, f"Best train Accuracy: {best_train_acc}", fontsize=12, color='blue', transform=plt.gca().transAxes)
plt.text(0.5, 0.43, f"Best validation Accuracy: {best_val_acc}", fontsize=12, color='red', transform=plt.gca().transAxes)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.text(0.5, 0.50, f"Best training loss: {best_train_loss}", fontsize=12, color='blue', transform=plt.gca().transAxes)
plt.text(0.5, 0.43, f"Best validation loss: {best_val_loss}", fontsize=12, color='red', transform=plt.gca().transAxes)
plt.legend()

plt.tight_layout()
plt.savefig('side-by-side accuracy-loss.png', dpi=300) # optional
plt.show()

########################### saving Section ###########################

