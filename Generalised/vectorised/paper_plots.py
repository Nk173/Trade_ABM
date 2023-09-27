
import numpy as np
from agents_vec import gulden_vectorised as gv
import time
import matplotlib.pyplot as plt
from functions import plot_gdp, plot_gdp_distribution, plot_sorted_production_matrix
np.random.seed(0)

## Case 0 - Base Gulden Model with Samuelson 
# case = 'Gulden_base_0'
# iterations = 2000
# n_countries = 2
# n_products = 2
# countries = ['USA', 'CHINA']
# products = ['wine', 'cloth']
# alpha = np.array([[0.5,0.5],
#                   [0.5,0.5]])
# beta  = np.array([[0.5,0.5],
#                   [0.5,0.5]])
# A     = np.array([[0.5, 2.0],
#                   [0.2, 0.05]])  # Total Factor Productivity
# shock = np.array([[0.5, 2.0],
#                   [0.2, 0.8]])  # Total Factor Productivity

# # Number of citizens in each nation
# citizens_per_nation = [100, 1000]
# t1 = time.time()
# gv(case = case, n_countries = n_countries, n_products=n_products, countries = countries, products = products, citizens_per_nation=citizens_per_nation, A=A, alpha=alpha, beta=beta,
#                   iterations=iterations, Tr_time=500, pricing_algorithm='dgp', utility_algorithm='geometric',
#                   shock =shock, shock_time=1000, cm_time=10000, plot=True)
# t2 = time.time()
# print('Vectorised Trade model time taken of {}: {} seconds'.format(case,t2-t1))

# # Constants
# iterations = 1000
# n_countries = 3
# countries = ['USA', 'China', 'India']
# products= ['wine', 'cloth', 'wheat']
# n_products = 3
# alpha = np.array([[0.5, 0.5,0.5],
#                   [0.5, 0.5,0.5],
#                   [0.5,0.5,0.5]])  # output elasticity of labor
# beta = np.array([[0.5, 0.5,0.5],
#                  [0.5, 0.5,0.5],
#                  [0.5, 0.5, 0.5]])  # output elasticity of capital
# A = np.array([[2, 1,1],
#               [1,2,1],
#               [1,1,2]])  # Total Factor Productivity
# shock = np.array([[0.5, 2.0],
#                   [0.2, 0.8]])  # Total Factor Productivity

# # Number of citizens in each nation
# citizens_per_nation = [100, 100, 100]
# t1 = time.time()
# gv(n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#                   iterations=2000, Tr_time=500, pricing_algorithm='cpmu', utility_algorithm='geometric',
#                   cap_mobility = True, shock =shock, shock_time=10000, cm_time=10000, plot=True)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))



# Constants
case = '2x2_GB_model'
iterations = 1000
n_countries = 2
countries = ['USA', 'Ghana']
products= ['wine', 'cloth']
n_products = 2
alpha = np.array([[0.7, 0.4],
                  [0.7, 0.4]])  # output elasticity of labor

beta = np.array([[0.7, 0.4],
                 [0.7, 0.4]])  # output elasticity of capital

A = np.array([[1, .5],
              [0.2, 0.2]])  # Total Factor Productivity

shock = np.array([[1, .5],
                  [1.5, 0.2]])  # Total Factor Productivity

# Number of citizens in each nation
citizens_per_nation = [500, 100]

t1 = time.time()
gv(case, n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
   iterations=3000, Tr_time=[500,2000], trade_change=0.05, autarky_time=1500, pricing_algorithm='dgp', utility_algorithm='geometric',
   wage_algorithm='marginal_product', shock = shock, shock_time=1000, cm_time=25000, plot=True, csv=False)
t2 = time.time()
print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))

# # 3x3 model
# case = '3x3_model'
# iterations = 1000
# n_countries = 3
# countries = ['USA', 'China', 'Ghana']
# products= ['wine','cloth', 'wheat']
# n_products = 3
# alpha = np.array([[0.5, 0.5, 0.5],
#                   [0.5, 0.5, 0.5],
#                   [0.5, 0.5, 0.5]])  # output elasticity of labor

# beta = np.array( [[0.5, 0.5, 0.5],
#                   [0.5, 0.5, 0.5],
#                   [0.5, 0.5, 0.5]])  # output elasticity of capital

# A = np.array([[1, 0.5, 2],
#               [1, 0.2, 0.05],
#               [1, 0.12, 0.03]])  # Total Factor Productivity

# shock = np.array([[1, 0.5, 2],
#                   [1, 0.2, 0.05],
#                   [1, 0.12, 0.03]])  # Total Factor Productivity
# # Number of citizens in each nation
# citizens_per_nation = [100, 100, 100]

# t1 = time.time()
# gv(case, n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#    iterations=3000, Tr_time=500, autarky_time=15000, pricing_algorithm='dgp', utility_algorithm='geometric',
#    wage_algorithm='share_of_product',
#     cap_mobility = False, shock = shock, shock_time=10000, cm_time=2500, plot=True, csv=False)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))

# # mxn model

# case = 'mxn_model'
# iterations = 1000
# n_countries = 20
# n_products = 50
# np.random.seed(0)
# # list of random names for conuntries and products
# import random
# import string
# countries = [''.join(random.choice(string.ascii_uppercase) for _ in range(5)) for _ in range(n_countries)]
# products = [''.join(random.choice(string.ascii_uppercase) for _ in range(5)) for _ in range(n_products)]

# # random matrix of alphas and betas
# alpha = np.random.rand(n_countries, n_products)
# beta = np.random.rand(n_countries, n_products)

# # random matrix of total factor productivity
# A = np.random.rand(n_countries, n_products)
# print(A)

# # matrix of ones for share
# share = np.ones((n_countries))
# # Number of citizens in each nation
# citizens_per_nation = [100 for _ in range(n_countries)]

# t1 = time.time()
# net_exports, income, production = gv(case, n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#    iterations=1000, Tr_time=500, autarky_time=15000, pricing_algorithm='cpmu', utility_algorithm='geometric', share=share,
#    wage_algorithm='marginal_product', shock = None, shock_time=10000, cm_time=2500, plot=False, csv=False,
#    innovation=True, innovation_time=750, gamma=1, eta=0.01)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))

# # Compute GDP values for each country
# gdp_values = income

# # Create dummy country names based on the number of countries
# country_names = ["Country {}".format(i+1) for i in range(n_countries)]

# # Use the plotting function to visualize the results
# plot_gdp_distribution(case, gdp_values)

# # Plot the production matrix and A matrix along side each other
# plot_sorted_production_matrix(case, A, production[:,:,-1])
# # np.savetxt("foo.csv", net_exports, delimiter=",")
