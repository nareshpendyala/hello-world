# Code: https://victorzhou.com/blog/intro-to-cnns-part-1/ #

import numpy as np

class Conv3x3:
    # A Convolution layer using 3x3 filters.

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # Filters is a 3d array with dimensions (num_filters, 3, 3)
        # Divide by 9 to reduce the variance of initial values
        self.filters = np.random.randn(num_filters, 3, 3)/9

        # Importance of value 9: If the initial values are too large or too small
        # training will be ineffective.
        # Concept: Xavier Initialization 
        # https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks

    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image region using valid padding.
        - image is a 2d numpy array
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j
    
    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        -input is a 2d numpy array
        '''
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i , j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1,2))
            
        return output