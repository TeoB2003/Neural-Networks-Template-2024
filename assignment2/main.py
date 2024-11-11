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


num_features = 784
num_classes=10

g_weights2=np.zeros((num_classes, num_features))
g_biases = np.zeros(num_classes)


learning_rate=0.05

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(train_x,train_y):
        learning_rate=0.01
        for x in range(len(train_x)):
            for i in range(10):
                if i==train_y[x]:
                    target=1
                else:
                    target=0
                classified=np.dot(train_x[x], g_weights2[i])+g_biases[i]
                output = sigmoid(classified) #fara sigmoid rata e de 70%
                error = target - output
                #print(f'{classified} si error {error}')
                g_weights2[i]+=error*learning_rate*train_x[x]
                g_biases[i]+=error*learning_rate


def train_all(epoch):
    for e in range(epoch):
        for i in range(0, len(train_X), 100):
            x_batch = np.array(train_X[i:i + 100])
            y_batch = train_Y[i:i + 100]
            train(x_batch,y_batch)
        test(e)

def test(epoch):
    nr_cases = 0
    good_cases = 0
    for i1 in range(len(test_X)):
        nr_cases += 1
        maxi = -20
        prediction = 0
        for k in range(10):
            rez = np.dot(test_X[i1], g_weights2[k]) + g_biases[k]
            if rez > maxi:
                maxi = rez
                prediction = k
        #print(f'Am prezis {prediction} si era {train_Y[i1]}')
        if prediction == test_Y[i1]:
            good_cases += 1
    accuracy = good_cases / nr_cases
    print(f'In epoca {epoch+1} am acuratete {accuracy*100:.2f}%')

train_all(1)
