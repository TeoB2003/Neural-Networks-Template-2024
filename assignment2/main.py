from http.client import error

import numpy as np
from sympy.physics.units import length
from torchvision.datasets import MNIST

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',transform=lambda x: np.array(x).flatten()/255.0,download=True,train=is_train)
    # impart la 255 pentru a normaliza datele
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return mnist_data, mnist_labels

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

def softmax(weights):
    Z_exp = np.exp(weights)
    softmax_output = Z_exp / np.sum(Z_exp, axis=1, keepdims=True) #impart fiecare element al lui Z_exp la suma
    return softmax_output

def cross_entropy_loss(probabilities, labels):
    log_predictions = np.log(probabilities + 1e-8)
    weighted_log_predictions = labels * log_predictions
    return np.mean(np.sum(weighted_log_predictions))

num_features = 784
num_classes=10

g_weights = np.zeros((num_features,num_classes))
g_biases = np.zeros(num_classes)

train_Y = np.eye(10)[train_Y]
test_Y = np.eye(10)[test_Y]

learning_rate=0.01

def train(batch_X, batch_Y):
    global g_biases, g_weights
    for i in range(0,len(batch_X)):
        class_matrix=np.dot(batch_X, g_weights)+ g_biases
        probabilities=softmax(class_matrix)
        loss = cross_entropy_loss(probabilities, batch_Y)
        gradient_loss = probabilities - batch_Y
        product = np.dot(batch_X.T, gradient_loss) /100
        for_bias = np.sum(gradient_loss, axis=0) / 100
        g_biases -= learning_rate * for_bias
        g_weights -= learning_rate * product

def compute_accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

def train_all(epochs):
    for epoch in range(epochs):
        for i in range(0, len(train_X), 100):
            x_batch = np.array(train_X[i:i + 100])
            y_batch = train_Y[i:i + 100]
            train(x_batch ,y_batch)
        train_accuracy = compute_accuracy(softmax(np.dot(train_X, g_weights) + g_biases), train_Y)
        test_accuracy = compute_accuracy(softmax(np.dot(test_X, g_weights) + g_biases), test_Y)
        print(f'Epoch {epoch + 1}  - Train Accuracy: {train_accuracy * 100:.2f}% - Test Accuracy: {test_accuracy * 100:.2f}%')
    final_train_accuracy = compute_accuracy(softmax(np.dot(train_X, g_weights) + g_biases), train_Y)
    final_test_accuracy = compute_accuracy(softmax(np.dot(test_X, g_weights) + g_biases), test_Y)
    print(f'Final Test Accuracy: {final_test_accuracy * 100:.2f}%')





train_all(2)


