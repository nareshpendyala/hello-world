import mnist
import numpy as np
from Convolution import Conv3x3
from Pooling import MaxPool2
from Softmax import Softmax

# First 1k testing examples are used out of 10k total
# The minst package handles the MNIST dataset for us
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = Conv3x3(8)                       # 28x28x1 => 26x26x8
pool = MaxPool2()                       # 26x26x8 => 13x13x8
softmax = Softmax(13 * 13 * 8, 10)      # 13x13x8 => 10

def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # Transform the image from [0, 255] to [-0.5, 0.5]
    output = conv.forward((image / 255) - 0.5)
    output = pool.forward(output)
    output = softmax.forward(output)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(output[label])
    acc = 1 if np.argmax(output) == label else 0

    return output, loss, acc

print('MNIST CNN initialized')

loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
    # A forward pass.
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

    # Print stats every 100 steps.
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i+1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0