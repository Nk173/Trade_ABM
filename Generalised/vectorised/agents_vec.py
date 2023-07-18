import numpy as np
import random
import sys
from functions import production_function, wage_function, demand_function
from tradeutils import doAllTrades
from pricing import updatePricesAndConsume

np.random.seed(0)

# Constants
iterations = 1000
n_countries = 2
countries = ['USA', 'China']
products= ['wine', 'cloth']
n_products = 2
alpha = np.array([[0.5, 0.5],
                  [0.5, 0.5]])  # output elasticity of labor
beta = np.array([[0.5, 0.5],
                 [0.5, 0.5]])  # output elasticity of capital
A = np.array([[0.5, 2],
              [0.2, 0.05]])  # Total Factor Productivity

# Number of citizens in each nation
citizens_per_nation = [100, 1000]

def gulden_vectorised(n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta, 
                      iterations=1000, Tr_time=500, pricing_algorithm='cpmu', utility_algorithm='geometric',
                      csv = False, plot = True):
    
    # Initialize prices for each good in each nation
    prices = np.ones((n_countries, n_products))
    prices_vec = np.ones((n_countries, n_products, iterations))

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
    S = np.zeros((n_countries, n_products))

    tv = np.zeros((n_countries, n_countries, n_products))
    labor = np.zeros((n_countries, n_products,iterations))
    capital = np.zeros((n_countries, n_products,iterations))
    production = np.zeros((n_countries, n_products,iterations))
    demand = np.zeros((n_countries, n_products,iterations))
    wage = np.zeros((n_countries, n_products,iterations))
    returns = np.zeros((n_countries, n_products,iterations))
    utility = np.zeros((n_countries, iterations))
    ############

    for t in range(iterations):
        Tr=False
        for nation in range(n_countries):
            # Initialise 
            should_change_labor = np.zeros(citizens_per_nation[nation])
            should_change_capital = np.zeros(citizens_per_nation[nation])

            # Citizen update
            # Re-assess labor and capital choices
            # Citizens decide whether to change their job or investment based on probability P
            max_wage_industry = np.argmax(W[nation,:])
            max_return_industry = np.argmax(R[nation,:])
            # print(W[nation,:], max_wage_industry, R[nation,:], max_return_industry)

            event_occurs = np.random.binomial(1, 0.004, size=citizens_per_nation[nation])
            should_change_labor = (event_occurs==1) * (W[nation, labor_choices[nation]]<= W[nation, max_wage_industry])
            should_change_capital = (event_occurs==1) * (R[nation, capital_choices[nation]]<= R[nation, max_return_industry])
            labor_choices[nation][should_change_labor==1] = max_wage_industry
            capital_choices[nation][should_change_capital==1] = max_return_industry
            income[nation] = sum(W[nation, labor_choices[nation]] + R[nation, capital_choices[nation]])

            for industry in range(n_products):
                L[nation,industry] = 0 
                K[nation, industry] = 0
                D[nation,industry] = income[nation]/(n_products*prices[nation,industry])
                demand[nation,industry,t] = D[nation,industry]
                L[nation,industry] = sum(labor_choices[nation] == industry)
                K[nation,industry] = sum(capital_choices[nation] == industry)
                if K[nation, industry]<1:
                    K[nation, industry] = 1

            # Calculate labor, capital, production, wages, and returns for each industry in each nation
            # for industry in range(n_products):
                W[nation, industry] = 0
                R[nation, industry] = 0
                Q[nation,industry] = production_function(A[nation,industry], L[nation,industry], K[nation,industry],alpha[nation,industry], beta[nation,industry])
                W[nation, industry] =  wage_function(A[nation,industry], L[nation,industry], K[nation,industry], alpha[nation,industry], beta[nation,industry],p=prices[nation, industry])
                R[nation, industry] = ((Q[nation,industry]*prices[nation, industry]) - (W[nation, industry]*L[nation,industry]))/K[nation,industry]
                S[nation, industry] = Q[nation, industry]

                # Containers
                labor[nation,industry,t] = L[nation,industry]
                capital[nation,industry,t] = K[nation,industry]
                production[nation,industry,t] = Q[nation,industry]
                wage[nation,industry,t] = W[nation, industry]
                returns[nation,industry,t] = R[nation, industry]

        # Trade update
        if t>=Tr_time:
            Tr=True
            trades = doAllTrades(tv, S, prices)
            net_exports = trades[1]
            S = Q - net_exports
            
        # vectorise prices
        prices, UT = updatePricesAndConsume(prices, D, S,pricing_algorithm, utility_algorithm)
        prices_vec[:,:,t] = prices
        utility[:,t] = UT

        # for nation in range(n_countries):
        #     for industry in range(n_products):

        #         # Citizen-Income, and Citizen-Consumption
        #         if industry>0 and (D[nation,industry] > S[nation,industry]):
        #             prices[nation,industry] = prices[nation,industry] + prices[nation,industry]*.02

        #         elif industry>0 and (D[nation,industry] < S[nation,industry]):
        #             prices[nation,industry] = prices[nation,industry]- prices[nation,industry]*0.02
        #         prices_vec[nation,industry,t] = prices[nation,industry]
                
    # Save the results of Labor, Productions, Wages, and Returns to a csv file
    if csv:
        import pandas as pd
        for nation in range(n_countries):
            for industry in range(n_products):
                df = pd.DataFrame({'t':range(iterations),
                                'labor':labor[nation,industry,:],
                                'capital':capital[nation,industry,:],
                                'production':production[nation,industry,:], 
                                'wages':wage[nation,industry,:],  
                                'returns':returns[nation,industry,:],
                                'prices':prices_vec[nation,industry,:],
                                'demand':demand[nation,industry,:],
                                'utility':utility[nation,:]})
                df.to_csv('csvs/{}_{}.csv'.format(countries[nation], products[industry]))
    
    # Plot the results
    if plot:
        import matplotlib.pyplot as plt
        variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI', 'MRS']
        fig,ax = plt.subplots(4,2)
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
                ax[2,0].plot(range(iterations), wage[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]))
                ax[2,0].set_title('Wages')
                ax[2,1].plot(range(iterations), returns[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]))
                ax[2,1].set_title('Returns')
                ax[3,0].plot(range(iterations), prices_vec[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]))
                ax[3,0].set_title('Prices')
                ax[3,1].plot(range(iterations), utility[nation,:], label='{}-{}'.format(countries[nation], products[industry]))
                ax[3,1].set_title('Utility')

        ax[3,0].legend()
        plt.show()

## model run
gulden_vectorised(n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
                  iterations=1000, Tr_time=500, pricing_algorithm='cpmu', utility_algorithm='geometric')
