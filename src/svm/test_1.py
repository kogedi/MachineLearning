import numpy as np
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


#optimization setup
objective = 

start = np.zeros(N)
bounds = [(0,C) for b in range(N)]
#bounds = [(0,None)] #To only have a lower bound
constraints = {'type':'eq','fun':zerofun}
    

#Solver
ret = minimize(objective, start, bounds, constraints=XC)
alpha = ret['x']