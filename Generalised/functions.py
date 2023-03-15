import random
import numpy as np
# Define the Cobb-Douglas production function
def production_function(A, alpha, L, beta, K):
    return A * (L ** alpha) * (K ** beta)

# Define the demand function
def demand_function(Y, P):
    
    D = np.zeros((len(P)))
    for p in range(len(P)):
        D[p] = Y / (2 * P[p])

    return D

# Define a Comparative Advantage function
# Define the RCA
def RCA(A,B):
    A = np.array(A)
    B = np.array(B)
    nA = A/A.sum()
    nB = B/B.sum()
    # print(nA,nB)
    return((nA>=nB)*1)