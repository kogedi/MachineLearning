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
#print("Kernel_matrix",kernel_matrix)

# IMPLEMENTATION
#
# 0 - setup variables
# 1 - objective
# 2 - start
# 3 - bounds
# 4 - constraints
# 5 - MINIMIZE (1, 2, 3, 4)
# 6 - resulting vector

# 0 - setup variables
alpha = [0.0] * N

# 1 - objective
def objective (alpha):
    obf = 0 
    alphas = 0
    for i in range(N):
        for j in range(N):
          obf = obf + alpha[i]*alpha[j]*kernel_matrix[i][j]
    for i in range(N):
        alphas = alphas + alpha[i]
    return  0.5 * obf - alphas 

#objective = lambda alpha: .. TODO Wie muss die Funktion strukturiert sein?


# 2 - start
start = numpy.zeros(N) #Why this start vector step-up. Maybe better np.zeros(N)
#print("Start",start)

# 3 - bounds
C = 1.0  #Slack variable C
B = [(0, C) for b in range(N)]

# 4 - constraints
#gleichung = 0 #TODO Wofür?
def zerofun(alpha, targets):
    # gleichung = 0.0
    # for i in range(N):
    #     gleichung =  alpha[i] * targets[i]
    gleichung = numpy.dot(alpha,targets)
    return gleichung 

zerofun = lambda alpha: numpy.dot(alpha,targets)
    
#constraints
XC = {'type':'eq', 'fun':zerofun},

# 5 - MINIMIZE (1, 2, 3, 4)
ret = minimize(objective, start, bounds=B, constraints = XC)

# 6 - resulting vector
alphamin = ret['x']

print("alphamin")
print(alphamin)
plt.plot(alphamin)


threshold = 1e-5  
non_zero_alphas = []
nz_x= []
nz_t= []

# TODO What is the part below doing?
for i in range(len(alphamin)):
    if alphamin[i] > threshold: 
        non_zero_alphas.append(alphamin[i])
        nz_x.append(inputs[i])
        nz_t.append(targets[i])
    
def calculate_b(alpha, targets, kernel_matrix, s):
    b_sum = 0.0
    for i in range(len(alpha)):
        b_sum += alpha[i] * targets[i] * kernel_matrix[s, i]
    b = b_sum - targets[s]
    return b

def indicator(alpha, targets, kernel_matrix, s):
    indicator_s = 0.0
    for i in range(N):
        indicator_s += alpha[i] * targets[i] * kernel_matrix[s, i]
    indi = indicator_s - calculate_b (alpha, targets, kernel_matrix, s)
    return indi

# 6 Plotting

#See the data
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

#6.1 Plotting the Decision Boundary
# xgrid=numpy. linspace (-5, 5)
# ygrid=numpy. linspace (-4, 4)

# grid=numpy. array ( [ [ indicator (x , y)
#                        for x in xgrid ]
#                      for y in ygrid ])

# plt . contour (xgrid , ygrid , grid ,
#                (-1.0, 0.0 , 1.0),
#                colors=('red', 'black' , 'blue' ),
#                linewidths =(1, 3 , 1))