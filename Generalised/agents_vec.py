<<<<<<< HEAD
=======
import numpy as np
import random

np.random.seed(1)

# Functions
def production_function(A, L, K, alpha, beta):
    return  A * (L**(alpha)) * (K**(beta))

# Constants
iterations = 500
n_countries = 2
countries = ['USA', 'China']
products= ['wine', 'cloth']
n_products = 2
alpha = 0.5  # output elasticity of labor
beta = 0.5  # output elasticity of capital
A = np.array([[0.5, 2],
              [0.2, 0.05]])  # Total Factor Productivity

# Number of citizens in each nation
citizens_per_nation = [100, 1000]

# Initialize prices for each good in each nation
prices = np.ones((n_countries, n_products))

# Initialize industry choices for labor and capital for each citizen in each nation
labor_choices = [np.zeros(citizens, dtype=int) for citizens in citizens_per_nation]
capital_choices = [np.zeros(citizens, dtype=int) for citizens in citizens_per_nation]
income = [np.zeros(citizens) for citizens in citizens_per_nation]
total_income = np.zeros(n_countries)

for nation in range(n_countries):
    labor_choices[nation] = np.random.choice(range(n_products), citizens_per_nation[nation])
    capital_choices[nation] = np.random.choice(range(n_products), citizens_per_nation[nation])

# Nation-level
# Calculate wages and returns to capital for each industry in each nation
L = np.zeros((n_countries, n_products))
K = np.zeros((n_countries, n_products))
Q = np.zeros((n_countries, n_products))
W = np.zeros((n_countries, n_products))
R = np.zeros((n_countries, n_products))
D = np.zeros((n_countries, n_products))
Tr = np.zeros((n_countries, n_products))
labor = np.zeros((n_countries, n_products,iterations))
capital = np.zeros((n_countries, n_products,iterations))
production = np.zeros((n_countries, n_products,iterations))
demand = np.zeros((n_countries, n_products,iterations))

for t in range(iterations):
    total_income = np.zeros(n_countries)
    for nation in range(n_countries):
        # Citizen update
        # Re-assess labor and capital choices
        # Citizens decide whether to change their job or investment based on probability P
        max_wage_industry = np.argmax(W[nation,:])
        max_return_industry = np.argmax(R[nation,:])

        event_occurs = np.random.binomial(1, 0.004, size=citizens_per_nation[nation])
        max_wage_industry = np.argmax(W[nation,:])
        max_return_industry = np.argmax(R[nation,:])
        
        should_change_labor = (event_occurs==1) * (W[nation, labor_choices[nation]]< W[nation, max_wage_industry])
        should_change_capital = (event_occurs==1) * (R[nation, capital_choices[nation]]<R[nation, max_return_industry])
        
        labor_choices[nation][should_change_labor==1] = max_wage_industry
        capital_choices[nation][should_change_capital==1] = max_return_industry
        income[nation] = W[nation, labor_choices[nation]] + R[nation, capital_choices[nation]]

        # Calculate labor, capital, production, wages, and returns for each industry in each nation
        for industry in range(n_products):
            L[nation,industry] = np.sum(labor_choices[nation] == industry)
            K[nation,industry] = np.sum(capital_choices[nation] == industry)
            Q[nation,industry] = production_function(A[nation,industry], L[nation,industry], K[nation,industry],alpha, beta)
            W[nation, industry] = production_function(A[nation,industry], L[nation,industry]+1, K[nation,industry],alpha, beta)-production_function(A[nation,industry], L[nation,industry], K[nation,industry],alpha, beta)
            R[nation, industry] = (Q[nation,industry]*prices[nation, industry] - W[nation, industry]*L[nation,industry])/K[nation,industry]
            labor[nation,industry,t] = L[nation,industry]
            capital[nation,industry,t] = K[nation,industry]
            production[nation,industry,t] = Q[nation,industry]
            # total_income[nation] = total_income[nation] + W[nation, industry]*L[nation,industry] + R[nation, industry]*K[nation,industry]   

    # Citizen-Income, and Citizen-Consumption
    for nation in range(n_countries):
        for industry in range(n_products):
            D[nation,industry] = sum(income[nation])/(n_products*prices[nation,industry])
            demand[nation,industry,t] = D[nation,industry]
            if industry>0 and (D[nation,industry] > Q[nation,industry]):
                prices[nation,industry] = prices[nation,industry]*1.02

            elif industry>0 and (D[nation,industry] < Q[nation,industry]):
                prices[nation,industry] = prices[nation,industry]*0.98
            
    
# Plot the results
import matplotlib.pyplot as plt
variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI', 'MRS']
fig,ax = plt.subplots(2,2)
for nation in range(n_countries):
    for industry in range(n_products):
        ax[0,0].plot(range(iterations), production[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]))
        ax[0,0].set_title('Production')
        ax[0,1].plot(range(iterations), demand[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]))
        ax[0,1].set_title('Demand')
        ax[1,0].plot(range(iterations), labor[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]))
        ax[1,0].set_title('Labor')
        ax[1,1].plot(range(iterations), capital[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]))
        ax[1,1].set_title('Capital')
plt.legend()
plt.show()
>>>>>>> 1a34f2d (vectorized branch - initial commit)
