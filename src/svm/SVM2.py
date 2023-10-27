"""
SVM Lab 2 of Claas and Konrad
"""  
        
import numpy, random, math 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from util import *

#******** SETUP VARIABLE PARAMETERS ***********
svm1 = SVM(kernel_choice='lin', exponent=2, sigma=3, slack_c=100) 

#Set Random to same values for every iteration
#numpy.random.seed(100)

# Different Data sets
#ini6 0,2 #ini5,0.8 #ini5,0.4   #ini4, 0.2   # ini 3      # ini 2          # ini 1
#Class A
                                #[1, 0.5]    #[1, 0.5]    #[1, 0.5]        #[1,5 0.5]
                                #[-1, 0.5]    #[-1, 0.5]   #[-1, 0.5]       #[-1.5, 0.5]
# Class B
                                #[0.0 , -1.5] #[0.0 , 0.5] #[0.0 , 0]       #[0.0 , -0.5]

# Data Generation
classA = numpy. concatenate ( 
    (numpy.random.randn(20, 2) * 0.2 + [1, 0.5],             
     numpy.random.randn(20, 2) * 0.2 + [-1, 0.5]))            

classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , -1.5] 

inputs = numpy. concatenate (( classA , classB ))
targets = numpy. concatenate ((numpy.ones(classA.shape[0]) , - numpy.ones ( classB.shape[ 0 ] ) ) ) 

N = inputs . shape [0] # Number of rows (samples)
numpy.random.seed(100)

#Permute the data arround 
permute = list (range(N)) 
random. shuffle (permute)
inputs = inputs [ permute , : ]
targets = targets [ permute ]


# Calculation of Kernelmatrix with ti and tj 
P_kernel_matrix = svm1.calculate_P_kernel_matrix(inputs,targets)
plt.matshow(numpy.log(numpy.abs(P_kernel_matrix)+1e-10))
plt.show()

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
# 10 - Plotting the Decision Boundary
# 11 - Safe Figure

#*****************************************************

# 0 - setup variables
alpha = numpy.ones(N)

# 1 - objective
# See Formula (4) objective = 0.5 * alpha.T * P *alpha - sum(alpha)
objective = lambda alpha: 0.5* numpy.dot(numpy.dot(alpha.T,P_kernel_matrix),alpha) - numpy.sum(alpha)

# 2 - start
start = numpy.ones(N) #Why this start vector step-up. Maybe better np.zeros(N)

# 3 - bounds
#Slack variable defined as a global variable of SVM
B = [(0, svm1.c) for b in range(N)]

# 4 - constraints
zerofun = lambda alpha: numpy.dot(alpha,targets)
#constraints
XC = {'type':'eq', 'fun':zerofun} #contraints zerofun to be ZERO

# 5 - MINIMIZE (1, 2, 3, 4)
ret = minimize(objective, start, bounds=B, constraints = XC)
#ret2 = minimize(objective, start, bounds=B, constraints = XC)
# 6 - resulting vector
alphamin = ret['x']

# 7 - plot alphamin

plt.plot(alphamin)
plt.plot(targets)
plt.show()

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
        
plt.plot(non_zero_alphas)
plt.plot(nz_t)
plt.show()
        
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
print('test erfolgreich')

# 10 - Plotting the Decision Boundary
xgrid=numpy. linspace (-5, 5)
ygrid=numpy. linspace (-4, 4)

grid=numpy. array ( [ [ svm1.indicator(non_zero_alphas, nz_t, nz_x, (x , y))
                       for x in xgrid ]
                     for y in ygrid ])

grid = numpy.array(grid)

plt.contour (xgrid , ygrid , grid ,
               (-1.0, 0.0 , 1.0),
               colors=('red', 'black' , 'blue' ),
               linewidths =(1, 3 , 1))

plt.show()

# 11 - Safe Pictures

import matplotlib.pyplot as plt

# Get the filename from user input
filename = input("Enter file name or No to abort without save: ")

# Save the plot with the provided filename
if filename == "No" or filename == "N":
    print("Not safed")
else:
    plt.savefig(f"{filename}.png")
    print(f"Plot saved as {filename}.png")