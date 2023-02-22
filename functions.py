# Define the Cobb-Douglas production function
def production_function(A, alpha, L, beta, K):
    return A * (L ** alpha) * (K ** beta)

# Define the demand function
def demand_function(Y, Pw, Pc):
    Dw = Y / (2 * Pw)
    Dc = Y / (2 * Pc)
    return Dw, Dc