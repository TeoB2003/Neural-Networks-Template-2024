import numpy as np
from torchvision.datasets import MNIST
from concurrent.futures import ThreadPoolExecutor

num_features = 784
num_hidden = 100
num_classes = 10
learning_rate = 0.03
epochs = 60
batch_size = 64


patience = 2
decay_factor = 0.8
min_learning_rate = 0.0001


def download_mnist(is_train: bool):
    dataset = MNIST(root='./data', transform=lambda x: np.array(x).flatten() / 255.0, download=True, train=is_train)
    mnist_data, mnist_labels = [], []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)


weights_input_hidden = np.random.randn(num_features, num_hidden) * 0.01
bias_hidden = np.zeros((1, num_hidden))
weights_hidden_output = np.random.randn(num_hidden, num_classes) * 0.01
bias_output = np.zeros((1, num_classes))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    k=np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - k)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Forward propagation
def forward_propagation(X):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = softmax(output_input)
    return hidden_input, hidden_output, output

# Backpropagation
def backpropagation(X, Y, hidden_input, hidden_output, output):
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

    Y_one_hot = np.eye(num_classes)[Y]
    error_output = output - Y_one_hot

    grad_weights_hidden_output = np.dot(hidden_output.T, error_output) / batch_size
    grad_bias_output = np.sum(error_output, axis=0, keepdims=True) / batch_size

    error_hidden = np.dot(error_output, weights_hidden_output.T) * sigmoid_derivative(hidden_input)

    grad_weights_input_hidden = np.dot(X.T, error_hidden) / batch_size
    grad_bias_hidden = np.sum(error_hidden, axis=0, keepdims=True) / batch_size

    weights_input_hidden -= learning_rate * grad_weights_input_hidden
    bias_hidden -= learning_rate * grad_bias_hidden

    weights_hidden_output -= learning_rate * grad_weights_hidden_output
    bias_output -= learning_rate * grad_bias_output

def calculate_accuracy(X, Y):
    _, _, output = forward_propagation(X)
    predictions = np.argmax(output, axis=1)
    return np.mean(predictions == Y) * 100


def process_batch(X_batch, Y_batch):
    hidden_input, hidden_output, output = forward_propagation(X_batch)
    backpropagation(X_batch, Y_batch, hidden_input, hidden_output, output)

def train_model(train_X, train_Y, test_X, test_Y):
    global learning_rate
    best_test_accuracy = 0
    no_improvement_epochs = 0

    for epoch in range(epochs):
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        train_X, train_Y = train_X[indices], train_Y[indices]

        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(train_X), batch_size):
                X_batch = train_X[i:i + batch_size]
                Y_batch = train_Y[i:i + batch_size]
                futures.append(executor.submit(process_batch, X_batch, Y_batch))

            for future in futures:
                future.result()

        train_accuracy = calculate_accuracy(train_X, train_Y)
        test_accuracy = calculate_accuracy(test_X, test_Y)
        print(f"Epoca {epoch + 1}/{epochs} - Acuratețe antrenare: {train_accuracy:.2f}% - Acuratețe validare: {test_accuracy:.2f}% - Rata de învățare: {learning_rate}")

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            learning_rate = max(learning_rate * decay_factor, min_learning_rate)
            print(f"Rata de învățare a fost redusă la {learning_rate}")
            no_improvement_epochs = 0

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

# Antrenare
train_model(train_X, train_Y, test_X, test_Y)
