import numpy, random, math 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds

class SVM:
    def __init__(self, kernel_choice, exponent, slack_c):
        """init of the Support Vector machine class

        Args:
            kernel_choice (str): Choose kernal 'lin', 'poly', 'rbf'
            slack_c (int): Slack value that is necessary for noisy data
        """
        if kernel_choice == 'lin':
            self.kernel = self.lin
        if kernel_choice == 'poly':
            self.kernel = self.poly
        if kernel_choice == 'rbf':
            self.kernel = self.rbf
        else:
            print("Fail: Choose vaild Kernel")
        self.c = slack_c #slack variable
        
        if exponent != None:
            self.exponent = exponent
        else:
            self.exponent = 2
        
    def poly(self, arrayx,arrayy):
        """generates a polynomial Kernal

        Args:
            arrayx (_array_): array of data
            arrayy (array): array of labels
            exponent (scalar): exponent, choose 2 or 3

        Returns:
            scalar: polynomial Kernel result
        """
        ploy = (numpy.dot(numpy.transpose(arrayx),arrayy)+1)**self.exponent
        return  ploy
    #def rbf(self,arrayx, arrayy):
    def lin(self, arrayx, arrayy):
        """linear Kernel

        Args:
            arrayx (_type_): _description_
            arrayy (_type_): _description_

        Returns:
            scalar: _description_
        """
        return numpy.dot(arrayx, arrayy)

    def calculate_P_kernel_matrix(self, data,targets):
        """calculates the P-kernel matrix, necessary for the objective function

        Args:
            data (array of [x,y]-entries): training data
            targets (vector): target labels to be learned

        Returns:
            _type_: prestep for Kernelmatrix called "Pij"
        """
        N = data.shape[0]
        P_kernel_matrix = numpy.zeros((N, N))

        for i in range(N):
            for j in range(N):
                P_kernel_matrix[i, j] = targets[i] * targets[j] * self.kernel(data[i], data[j])

        return P_kernel_matrix


    def calculate_b(self, non_zero_alpha, non_zero_targets, non_zero_data, s):
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
        
        #kernel_vector = calculate_kernel_vector(s, non_zero_data)
        b_sum = 0.0
        for i in range(len(non_zero_alpha)):
            b_sum += non_zero_alpha[i] * non_zero_targets[i] *self.kernel(non_zero_data[sv], non_zero_data[i])
        b = b_sum - non_zero_targets[sv]
        return b

    def indicator(self, alpha, targets, data, s):
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
            indicator_s += alpha[i] * targets[i] * self.kernel(s, data[i])
        indi = indicator_s - self.calculate_b(alpha, targets, data, s) #TODO How to implement the threshold correctly?
        return indi

    # def ind(self, datap, data):
    #     length = len(data)
    #     s = 0

    #     for i in range(length):
    #         v = usedKernel(datap, data[i][1][:-1], dimen)
    #         s = s + data[i][0] * data[i][1][-1] * v
    #     return s