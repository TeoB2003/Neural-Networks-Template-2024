import numpy as np
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

num_features = 784
num_classes=10

g_weights = np.zeros((num_classes, num_features))
g_biases = np.zeros(num_classes)

def train(x_batch, y_batch):
    global g_weights, g_biases
    learning_rate = 0.001
    weights=np.zeros((num_classes, num_features))
    bias=np.zeros(num_classes)
    for x in range(len(x_batch)):
        digit=y_batch[x]
        classified = np.dot(x_batch[x],g_weights[digit])+ g_biases[digit]
        #print(f'am clasifcat ca {classified} desi era {y_batch[x]}')


        error = digit - classified
        weights[digit] += + (x_batch[x] * learning_rate * error)
        bias[digit]+=learning_rate*error*digit
    return weights, bias


def train_all(train_X, train_Y, epochs):
    global g_weights, g_biases

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    for epoch in range(epochs):
        #p = np.random.permutation(len(train_X))
        #train_X = train_X[p]
        #train_Y=train_Y[p]
        """
        num_batches = len(train_X) // 100
        for batch in range(num_batches):
            start = batch * 100
            end = start + 100
            x_batch = train_X[start:end]
            y_batch = train_Y[start:end]

            weights, bias = train(x_batch, y_batch)

            g_weights += weights
            g_biases += bias


        if len(train_X) % 100 != 0:
            x_batch = train_X[num_batches * 100:]
            y_batch = train_Y[num_batches * 100:]
            weights, bias = train(x_batch, y_batch)
            g_weights += weights
            g_biases += bias
        """
        l_weights,l_biases=train(train_X,train_Y)
        g_biases+=l_biases
        g_weights+=l_weights


def test_model(test_X, test_Y):
    good_case=0
    total_cases=0
    for x in range(0,len(test_X)):
        total_cases+=1
        prediction=0
        diff_min=10
        for i in range(0,10):
            diff=(np.dot(g_weights[i], test_X[x])+ g_biases[i])-i
            print(f'diff pentru {i} este {diff}')
            if diff_min>diff:
                diff_min=diff
                prediction=i

        print(f'am prezis {prediction} si era {test_Y[x]}')
        if prediction==test_Y[x]:
            good_case+=1

    accuracy=good_case/total_cases
    return accuracy


# Example usage
train_all(train_X,train_Y,2)

accuracy = test_model(test_X, test_Y)  # Test the model
print(f'Test accuracy: {accuracy * 100:.2f}%')