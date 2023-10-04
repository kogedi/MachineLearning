import numpy, random, math 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds


classA = numpy. concatenate (
    (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
     numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , -0.5]


inputs = numpy. concatenate (( classA , classB ))
targets = numpy. concatenate ((numpy.ones(classA.shape[0]) , - numpy.ones ( classB.shape[ 0 ] ) ) ) 

N = inputs . shape [0] # Number of rows (samples)

permute = list (range(N)) 
random. shuffle (permute)
inputs = inputs [ permute , : ]
targets = targets [ permute ]


def kernel (arrayx, arrayy):
    return numpy.dot(arrayx, arrayy)

def compute_kernel_matrix(data, kernel_function):

    kernel_matrix = numpy.zeros((N, N))

    for i in range(N):
        for j in range(N):
            kernel_matrix[i, j] = targets[i]*targets[j]*kernel_function(data[i], data[j])

    return kernel_matrix #pij

# Berechnung der Kernelmatrix schon mit ti und tj 
kernel_matrix = compute_kernel_matrix(inputs, kernel)

# Ausgabe der Kernelmatrix
print(kernel_matrix)


def objective (alpha):
    return  0.5* numpy.sum(numpy.sum(alpha*alpha*kernel_matrix, axis=0),axis=1) - numpy.sum(alpha,axis=1)

def gleichung (alpha):
    numpy.sum(alpha*targets, axis=1)

constraint={'type':'eq', 'fun':gleichung},

# Beschränkung alpha
C = 1.0  
alpha_bounds = Bounds(0, C)

ret = minimize(objective, 0, bounds = alpha_bounds, constraints=constraint)

alpha = ret['x']

threshold = 1e-5  
almost_zero_alphas = []
for alpha_i in alpha:
    if alpha_i < threshold:
        almost_zero_alphas.append(alpha_i)

def calculate_b(alpha, targets, kernel_matrix, s):
    b_sum = 0.0
    for i in range(len(alpha)):
        b_sum += alpha[i] * targets[i] * kernel_matrix[s, i]

    # Subtrahieren Sie t_s
    b = b_sum - targets[s]
    return b

def indicator(alpha, targets, kernel_matrix, s):
    indicator_s = 0.0
    for i in range(N):
        indicator_s += alpha[i] * targets[i] * kernel_matrix[s, i]

    # Subtrahieren Sie t_s
    indi = indicator_s - calculate_b (alpha, targets, kernel_matrix, s)
    return indi

plt.plot([p[0] for p in classA] ,
         [p[1] for p in classA] ,
         'b.') 
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.') 

plt.axis('equal') # Force same scale on both axes 
plt.savefig('svmplot.pdf') # Save a copy in a file
plt .show() # Show the plot on the screen
print('test')