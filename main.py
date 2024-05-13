import numpy  as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Step 1: Load the data
def load_data(filepath):
    # Load the data from the text file
    data = np.loadtxt(filepath)
    labels = np.repeat(np.arange(10), 200)  # 200 examples per digit class
    return data, labels

# Step 2: Preprocess the data
def preprocess_data(data, labels):
    # Normalize the pixel values from 0-6 to 0-1
    data = data / 6.0
    # Reshape data to fit the model (batch_size, height, width, channels)
    data = data.reshape(-1, 15, 16, 1)
    # Convert labels to one-hot encoding
    labels = to_categorical(labels)
    return data, labels

# Step 3: Split the data
def split_data(data, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, random_state=42, stratify=labels)
    return X_train, X_test, y_train, y_test

# Step 4: Build the CNN model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.8), #it was 0.5
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 5: Train the model
def train_model(model, X_train, y_train, epochs=10):
    model.fit(X_train, y_train, epochs=epochs, verbose=1)
    return model

# Step 6: Evaluate the model
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    test_error_rate = (1 - test_acc) * 100  # Calculate error rate and convert to percentage
    print(f'Test accuracy: {test_acc:.3f}')
    print(f'Test loss: {test_loss:.3f}')
    print(f'Test error rate: {test_error_rate:.2f}%')  # Print error rate as a percentage



# Main Execution Flow
if __name__ == "__main__":
    data_path = 'mfeat-pix.txt'
    data, labels = load_data(data_path)
    data, labels = preprocess_data(data, labels)
    X_train, X_test, y_train, y_test = split_data(data, labels)
    model = build_model((15, 16, 1))  # Image dimensions and 1 for grayscale
    model = train_model(model, X_train, y_train, epochs=50)  # Increase epochs for better training
    evaluate_model(model, X_test, y_test)
