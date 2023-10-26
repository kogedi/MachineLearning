import numpy as np
from scipy import misc
from labfuns import *

    
# # X,y =  genBlobs(centers=5) #X, labels =
# # classes = np.unique(y) # Get the unique examples
# # # Iterate over both index and value
# # for jdx,clas in enumerate(classes):
# #     idx = y==clas # Returns a true or false with the length of y

# #     # Or more compactly extract the indices for which y==class is true,
# #     # analogous to MATLAB’s find
# #     idx = np.where(y==clas)[0]
# #     xlc = X[idx,:] # Get the x for the class labels. Vectors are rows.

# # Subtract the vector using for loop
# X = np.array([(1,2,3),(4,5,6),(7,8,9)])
# u = np.array((1,2,3))
# for i_row in range(0,X.shape[0]):
#     X[i_row,:] = X[i_row,:] - u
# print(X)
# # >> array([[0, 0, 0],
# # [3, 3, 3],
# # [6, 6, 6]])

# # Subtract using broadcasting
# X = np.array([(1,2,3),(4,5,6),(7,8,9)])
# u = np.array((1,2,3))
# X = X - u
# print(X)
# # >> array([[0, 0, 0],
# # [3, 3, 3],
# # [6, 6, 6]])