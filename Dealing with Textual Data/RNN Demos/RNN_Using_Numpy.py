import copy, numpy as np
np.random.seed(0)

# Sigmoid Activation Function
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# Derivative of Sigmoid Activation Function
def sigmoid_output_to_derivative(output):
    return output*(1-output)

# Binary Addition

# Generating dataset for training and testing
int2binary = {} # List of mapping of integer value and its binary representation
binary_dim = 8 # Number for 2 raised to the power value to find Integer values
no_of_bits = 8 # Number of bits considered in the binary representation

largest_number = pow(2,binary_dim) # Maximum integer value to be considered (2^8 = 256)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T, axis = 1) # Convert the integer from 0 to largest number into binary representation
for i in range(largest_number):
    int2binary[i] = binary[i] # Save the mapping of integer value and its binary mapping

# Hyperparameters required to train the neural network
alpha = 0.1 # Learning rate
input_dim = 2 # Since at each time step 1 bit from each number will be considered therefore input dimension is 2, i.e., 1 bit from each number considered for addition
hidden_dim = 16 # Dimension of Feature array that represent the memory state
output_dim = 1 # Output at each time step

# Initialization of Neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 # Input and Memory [2 * 16]
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 # Memory and Output [16 * 1]
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # Between memory of two time steps [16 * 16]

# Initialise the updated value of Neural network weights to zero
synapse_0_update = np.zeros_like(synapse_0) 
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# [1 * 2](Input at first time step) --[2 * 16](Synapse_0)---> Memory [1 * 16](Feature matrix representation the state of memory) ---[16 * 1](Synapse_1)---> Output [1 * 1]
# [1,1]
#                                                                        | 
#                                                                        | Synapse_h[16 * 16]
#                                                                        | 
#                                                                        V
# [1 * 2](Input at second time step) --[2 * 16](Synapse_0)---> Memory [1 * 16](Feature matrix representation the state of memory) ---[16 * 1](Synapse_1)---> Output [1 * 1]
# [1,1]

# Training of the Recurrent Neural Network (RNN)
# Simple Binary Addition Problem (a + b = c)
for j in range(10000):
    
    a_int = np.random.randint(largest_number/2) # Randomly select any integer value between 0 and largest_number/2
    a = int2binary[a_int] # Binary representation of the number selected
    # For example: a_int = 9, a = 00001001

    b_int = np.random.randint(largest_number/2) # Randomly select any integer value between 0 and largest_number/2
    b = int2binary[b_int] # Binary representation of the number selected
    # For example: b_int = 5, b = 00000101

    # Actual Output value
    c_int = a_int + b_int # Adding the integer values selected above
    c = int2binary[c_int] # Binary representation of the output after adding the two integer values
    # c_int = 14, c = 00001110
    # Store the predicted output at each time step in "d"
    d = np.zeros_like(c)

    overallError = 0 # Sum Error at each time step to calculate the total error loss
    
    layer_2_deltas = list() # Change in the layer_2 value
    layer_1_values = list() # List of feature vector representing Memory at each step
    layer_1_values.append(np.zeros(hidden_dim))
    
    # Forward Propagation going from right to left in the binary addition
    for position in range(binary_dim):
        
        # Selection of bits from each binary representation of integer values
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]]) # For first iteration: X = [1,0]
        y = np.array([[c[binary_dim - position - 1]]]).T # For first iteration: y = [1]

        # New feature vector representation of the memory
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # Output value after each time step
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # Calculate error loss at each time step
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2)) # Change in the error loss = Magnitude * direction of movement
        overallError += np.abs(layer_2_error[0])
    
        # Actual value in terms of binary value at each time step
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # Store memory state feature values to use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    # Backpropogation through time (BPTT)
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # Error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # Error at memory layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # Update weight values for the next iteration of the time step
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    if(j % 1000 == 0):
        print ("Error:" + str(overallError))
        print ("Pred:" + str(d))
        print ("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print (str(a_int) + " + " + str(b_int) + " = " + str(out))
        print ("------------")