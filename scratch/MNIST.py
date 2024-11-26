import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load training data
data_train = pd.read_csv('train.csv')

data_train = np.array(data_train)
m_train, n_train = data_train.shape
np.random.shuffle(data_train)  # shuffle the training data

# Prepare training set
data_train = data_train.T
Y_train = data_train[0]
X_train = data_train[1:n_train]
X_train = X_train / 255.0  # normalize

# Load test data (as dev set)
data_dev = pd.read_csv('test.csv')
data_dev = np.array(data_dev)
m_dev, n_dev = data_dev.shape

# Prepare dev set
data_dev = data_dev.T
Y_dev = data_dev[0]
X_dev = data_dev[1:n_dev]
X_dev = X_dev / 255.0  # normalize

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m_train):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m_train * dZ2.dot(A1.T)
    db2 = 1 / m_train * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m_train * dZ1.dot(X.T)
    db1 = 1 / m_train * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def compute_loss(A2, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    return -np.sum(one_hot_Y * np.log(A2)) / m

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X_train, Y_train, X_dev, Y_dev, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    m_train = X_train.shape[1]  # This gives you the number of training samples

    for i in range(iterations + 1):
        # Forward prop on training set
        Z1_train, A1_train, Z2_train, A2_train = forward_prop(W1, b1, W2, b2, X_train)
        
        # Backprop and update parameters (pass m_train here)
        dW1, db1, dW2, db2 = backward_prop(Z1_train, A1_train, Z2_train, A2_train, W1, W2, X_train, Y_train, m_train)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Calculate training loss and accuracy
        train_loss = compute_loss(A2_train, Y_train)
        train_accuracy = get_accuracy(get_predictions(A2_train), Y_train)
        
        # Forward prop on validation set
        Z1_dev, A1_dev, Z2_dev, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
        val_loss = compute_loss(A2_dev, Y_dev)
        val_accuracy = get_accuracy(get_predictions(A2_dev), Y_dev)
        
        # Store losses and accuracies for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Print progress every 10 iterations
        if i % 10 == 0:
            print(f"Iteration: {i}")
            print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return W1, b1, W2, b2, train_losses, val_losses, train_accuracies, val_accuracies

W1, b1, W2, b2, train_losses, val_losses, train_accuracies, val_accuracies = gradient_descent(X_train, Y_train, X_dev, Y_dev, 0.10, 5_000)

# If the training loss is decreasing but the validation loss is increasing, this is a key sign of overfitting
# Stop training if this is occuring

# Plot training and validation loss
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show(block=False)

# Create the second plot for training and validation accuracy
plt.figure()
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2, total_tests):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    
    current_image = current_image.reshape((28, 28)) * 255
    
    plt.figure()
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    
    plt.title(f"Predicted: {prediction[0]}, Actual: {label}")
    
    if index == total_tests - 1:
        plt.show()
    else:
        plt.show(block=False)

total_tests = 4

for i in range(total_tests):
    test_prediction(i, W1, b1, W2, b2, total_tests)

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print("Validation Accuracy: " + str(get_accuracy(dev_predictions, Y_dev)))