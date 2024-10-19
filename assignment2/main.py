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

learning_rate=0.001
num_features = 784
weights = np.zeros(num_features + 1)

def train(nr:int):
  for i in range(0,3):
    global weights
    global learning_rate
    total_number=0
    good_case=0
    for x in range(0,len(train_X)):
            total_number+=1
            input_with_bias = np.append(train_X[x], 1)
            classified= np.dot(weights, input_with_bias)
            #print(f'clasificat ca {classified} desi este {train_Y[x]} ')
            target = 1 if train_Y[x] == nr else -1

            if classified*target>0 :
                good_case+=1
            else:
                error = target - classified
                weights=weights+(input_with_bias*learning_rate*error)
                #print('Wrong!')
                #as putea sa ascot if-ul, dar il pastrez pentru claritate
                #trebuie folosite batchuri


def test():
    global weights
    total_test_samples = len(test_X)
    correct_predictions = np.zeros(10)  # Array to hold correct predictions for each digit
    total_per_digit = np.zeros(10)       # Array to hold total samples for each digit
    total_correct = 0                     # Counter for overall correct predictions

    for x in range(total_test_samples):
        input_with_bias = np.append(test_X[x], 1)
        classified = np.dot(weights, input_with_bias)
        prediction = 1 if classified > 0 else 0

        actual_digit = test_Y[x]  # Actual label of the current test sample
        total_per_digit[actual_digit] += 1  # Increment total count for the actual digit

        if (prediction == 1 and actual_digit == 1) or (prediction == 0 and actual_digit != 1):
            correct_predictions[actual_digit] += 1  # Increment the correct prediction for the actual digit
            total_correct += 1  # Increment the overall correct predictions

    # Calculate and print accuracy for each digit
    for digit in range(10):
        if total_per_digit[digit] > 0:  # Avoid division by zero
            accuracy = correct_predictions[digit] / total_per_digit[digit]
            print(f'Accuracy on test set for digit {digit}: {accuracy * 100:.2f}%')
        else:
            print(f'No samples for digit {digit} in the test set.')

    # Calculate and print overall accuracy
    overall_accuracy = total_correct / total_test_samples
    print(f'Overall accuracy on test set: {overall_accuracy * 100:.2f}%')

for i in range(0,10):
    train(i)

test()