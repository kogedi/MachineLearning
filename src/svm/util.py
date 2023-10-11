import numpy, random, math 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds

def kernel (arrayx, arrayy):
    """linear Kernel

    Args:
        arrayx (_type_): _description_
        arrayy (_type_): _description_

    Returns:
        scalar: _description_
    """
    return numpy.dot(arrayx, arrayy)

def calculate_P_kernel_matrix(data,targets):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: prestep for Kernelmatrix called "Pij"
    """
    N = data.shape [0]
    P_kernel_matrix = numpy.zeros((N, N))

    for i in range(N):
        for j in range(N):
            P_kernel_matrix[i, j] = targets[i]*targets[j]*kernel(data[i], data[j])

    return P_kernel_matrix

def calculate_kernel_vector(s, data):
    N = len(data)
    kernel_vector = []  
    for i in range(N):
            kernel_vector.append(kernel(s, data[i]))
    return kernel_vector 


def calculate_b(non_zero_alpha, non_zero_targets, non_zero_data, s):
    """calculates the scalar threshold value b

    Args:
        alpha (vector): optimized values of alpha (decision variable)
        targets (vector): target values for data points
        kernel_matrix (matrix): already calculates Kernel matrix to safe computation
        s (vector): new data vector to be classified

    Returns:
        b (scalar): threshold value b
    """
    #Choose one support vector out of the list
    sv = 1
    
    kernel_vector = calculate_kernel_vector(s, non_zero_data)
    b_sum = 0.0
    for i in range(len(non_zero_alpha)):
        b_sum += non_zero_alpha[i] * non_zero_targets[i] *kernel_vector[i]
    b = b_sum - non_zero_targets[sv]
    return b

def indicator(alpha, targets, data, s):
    """classification of a new data point s

    Args:
        alpha (vector): optimal decision variables
        targets (vector): labels of the training data
        data (matrix): precalculated Kernel matrix
        s (vectors): new datapoint

    Returns:
        scalar: indicates the class [-1, 0, +1]
    """
    indicator_s = 0.0
    for i in range(len(data)):
        indicator_s += alpha[i] * targets[i] * kernel(s, data[i])
    #indi = indicator_s - calculate_b (alpha, targets, data, s)
    return indicator_s

def ind(datap, data):
	length = len(data)
	s = 0

	for i in range(length):
		v = usedKernel(datap, data[i][1][:-1], dimen)
		s = s + data[i][0] * data[i][1][-1] * v
	return s