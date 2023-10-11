import numpy, random, math 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from util import *

#Set Random to same values for every iteration
numpy.random.seed(100)

# Data Generation
classA = numpy. concatenate (
    (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
     numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))

#print("classA",classA)
classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , -0.5]

inputs = numpy. concatenate (( classA , classB ))
targets = numpy. concatenate ((numpy.ones(classA.shape[0]) , - numpy.ones ( classB.shape[ 0 ] ) ) ) 

N = inputs . shape [0] # Number of rows (samples)

permute = list (range(N)) 
random. shuffle (permute)
inputs = inputs [ permute , : ]
targets = targets [ permute ]


# Berechnung der Kernelmatrix schon mit ti und tj 
#P_matrix = pre_calculate_P_kernel(inputs, kernel)
kernel_matrix = calculate_P_kernel_matrix(inputs,targets)

# Ausgabe der Kernelmatrix
#print("Kernel_matrix",kernel_matrix)

#************ IMPLEMENTATION Outline ******************
#
# 0 - setup variables
# 1 - objective
# 2 - start x0
# 3 - bounds
# 4 - constraints
# 5 - MINIMIZE function (1, 2, 3, 4)
# 6 - resulting vector
# 7 - plot alphamin
# 8 - Separate non-zero alpha-Values and plot
# 9 - Plotting decision boundaries

# 0 - setup variables
alpha = numpy.zeros(N)

# 1 - objective
# See Formula (4) 0.5 * alpha.T * P *alpha - sum(alpha)
objective = lambda alpha: 0.5* numpy.dot(numpy.dot(alpha.T,kernel_matrix),alpha) - numpy.sum(alpha)

# 2 - start
start = numpy.ones(N) #Why this start vector step-up. Maybe better np.zeros(N)

# 3 - bounds
c = 1.0  #Slack variable C
B = [(0, c) for b in range(N)]

# 4 - constraints
zerofun = lambda alpha: numpy.dot(alpha,targets)
#constraints
XC = {'type':'eq', 'fun':zerofun} #contraints zerofun to be ZERO

# 5 - MINIMIZE (1, 2, 3, 4)
ret = minimize(objective, start, bounds=B, constraints = XC)

# 6 - resulting vector
alphamin = ret['x']

# 7 - plot alphamin
#print("alphamin")
#print(alphamin)
# print("data")
# print(inputs)
# plt.plot(alphamin)
# plt.plot(targets)
# plt.show()

# 8 - Separate non-zero alpha-Values and plot
threshold = 1e-5  
non_zero_alphas = []
nz_x= []
nz_t= []

for i in range(len(alphamin)):
    if alphamin[i] > threshold: 
        non_zero_alphas.append(alphamin[i])
        nz_x.append(inputs[i])
        nz_t.append(targets[i])
        
# plt.plot(non_zero_alphas)
# plt.plot(nz_t)
# plt.show()
# print("Nonzero data")
# print(nz_x)
        
# 9 - Plotting decision boundaries
plt.plot([p[0] for p in classA] ,
         [p[1] for p in classA] ,
         'b.') 
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.') 

plt.axis('equal') # Force same scale on both axes 
plt.savefig('svmplot.pdf') # Save a copy in a file
#plt .show() # Show the plot on the screen
print('test')


# 10 - Plotting the Decision Boundary
xgrid=numpy. linspace (-5, 5)
ygrid=numpy. linspace (-4, 4)

grid=numpy. array ( [ [ indicator(non_zero_alphas, nz_t, nz_x, (x , y))
                       for x in xgrid ]
                     for y in ygrid ])

grid = numpy.array(grid)

plt.contour (xgrid , ygrid , grid ,
               (-1.0, 0.0 , 1.0),
               colors=('red', 'black' , 'blue' ),
               linewidths =(1, 3 , 1))
plt.show()