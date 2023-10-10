import numpy as np
import numpy
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Parameter and decision variables
N = 20 #number of data points
C = 5 #slack
#alpha
#t

#def objectivefun(alpha, )

zerofun = np.dot(alpha,t)#Summe(alpha_i * t_i) = 0

#Gernerate Test Data
classA = numpy.concatenate(
(numpy.random. randn(10 , 2) * 0.2 + [1.5 , 0.5] ,
numpy.random. randn(10 , 2) * 0.2 + [ -1.5 , 0.5]))

classB = numpy.random. randn(20 , 2) * 0.2 + [0.0 , -0.5]

inputs = numpy. concatenate (( classA , classB ))
targets = numpy. concatenate(
(numpy. ones ( classA . shape [0]) ,
- numpy.ones ( classB . shape [0])))
N = inputs . shape [0] # Number o f rows ( s a m p l e s )
permute = list(range (N))
random. shuffle (permute)
inputs = inputs [ permute , : ]
targets = targets [ permute ]


#optimization setup
#objective = 

start = np.zeros(N)
bounds = [(0,C) for b in range(N)]
#bounds = [(0,None)] #To only have a lower bound
constraints = {'type':'eq','fun':zerofun}

#Solver
ret = minimize(objective, start, bounds, constraints=XC)
alpha = ret['x']


#Plot

plt . plot ([p [0] for p in classA ] ,
[p [1] for p in classA ] ,
'b. ' )
plt . plot ([p [0] for p in classB ] ,
[p [1] for p in classB ] ,
' r . ' )
plt . axis (' equal ') # Force same s c a l e on b o t h a x e s
plt . savefig ( 'svmplot . pdf ' ) # Save a copy i n a f i l e
plt . show() # Show t h e p l o t on t h e s c r e e n