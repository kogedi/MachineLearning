import numpy, random, math 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds

numpy.random.seed(100)

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
    """linear Kernel

    Args:
        arrayx (_type_): _description_
        arrayy (_type_): _description_

    Returns:
        scalar: _description_
    """
    return numpy.dot(arrayx, arrayy)

def pre_calculate_P_kernel(data, kernel_function):
    """_summary_

    Args:
        data (_type_): _description_
        kernel_function (_type_): _description_

    Returns:
        _type_: prestep for Kernelmatrix called "Pij"
    """

    kernel_matrix = numpy.zeros((N, N))

    for i in range(N):
        for j in range(N):
            kernel_matrix[i, j] = targets[i]*targets[j]*kernel_function(data[i], data[j])

    return kernel_matrix 

# Berechnung der Kernelmatrix schon mit ti und tj 
#P_matrix = pre_calculate_P_kernel(inputs, kernel)
kernel_matrix = pre_calculate_P_kernel(inputs, kernel)

# Ausgabe der Kernelmatrix
#print("Kernel_matrix",kernel_matrix)

#************ IMPLEMENTATION ******************
#
# 0 - setup variables
# 1 - objective
# 2 - start x0
# 3 - bounds
# 4 - constraints
# 5 - MINIMIZE function (1, 2, 3, 4)
# 6 - resulting vector

# 0 - setup variables
alpha = numpy.zeros(N) #[0.0] * N

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
c = 1.0  #Slack variable C
B = [(0, c) for b in range(N)]

# 4 - constraints
#gleichung = 0 #TODO Wofür?
# def zerofun(alpha, targets):
#     # gleichung = 0.0
#     # for i in range(N):
#     #     gleichung =  alpha[i] * targets[i]
#     gleichung = numpy.dot(alpha,targets)
#     return gleichung 

zerofun = lambda alpha: numpy.dot(alpha,targets)
    
#constraints
XC = {'type':'eq', 'fun':zerofun} #contraints zerofun to be ZERO

# 5 - MINIMIZE (1, 2, 3, 4)
ret = minimize(objective, start, bounds=B, constraints = XC)

# 6 - resulting vector
alphamin = ret['x']

print("alphamin")
print(alphamin)
plt.plot(alphamin)
plt.show()


threshold = 1e-5  
non_zero_alphas = []
nz_x= []
nz_t= []

# Separate non-zero alpha-Values

for i in range(len(alphamin)):
    if alphamin[i] > threshold: 
        non_zero_alphas.append(alphamin[i])
        nz_x.append(inputs[i])
        nz_t.append(targets[i])
        
      

def calculate_kernel_vector(s,inputs,kernel_function):
    kernel_vektor = []  
    for i in range(N):
            kernel_vektor.append(kernel_function(s, inputs[i]))
    return kernel_vektor          
    
def calculate_b(alpha, targets, inputs, s,kernel_function):
    """calculates the threshhold value b

    Args:
        alpha (vector): optimized values of alpha (decision variable)
        targets (vector): target values for data points
        kernel_matrix (matrix): already calculates Kernel matrix to safe computation
        s (vector): new data vector to be classified

    Returns:
        scalar: offset of the hyperplane (here: 2-d case)
    """
    
    b_sum = 0.0
    for i in range(len(alpha)):
        b_sum += alpha[i] * targets[i] * calculate_kernel_vector(s,inputs,kernel_function)[i]
    b = b_sum - targets[s]
    return b

def indicator(alpha, targets, kernel_matrix, s,kernel_function):
    """classification of a new data point s

    Args:
        alpha (vector): optimal decision variables
        targets (vector): labels of the training data
        kernel_matrix (matrix): precalculated Kernel matrix
        s (_type_): new datapoint

    Returns:
        _type_: _description_
    """
    indicator_s = 0.0
    for i in range(N):
        indicator_s += alpha[i] * targets[i] * calculate_kernel_vector(s,inputs,kernel_function)[i]
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