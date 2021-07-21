import mnist
from Convolution import Conv3x3
from Pooling import MaxPool2
from Softmax import Softmax

# The minst package handles the MNIST dataset for us
train_images = mnist.train_images()
train_labels = mnist.train_labels()

conv = Conv3x3(8)                       # 28x28x1 => 26x26x8
pool = MaxPool2()                       # 26x26x8 => 13x13x8
softmax = Softmax(13 * 13 * 8, 10)      # 13x13x8 => 10

output = conv.forward(train_images[0])
output = pool.forward(output)
output = softmax.forward(output)

print(output.shape)