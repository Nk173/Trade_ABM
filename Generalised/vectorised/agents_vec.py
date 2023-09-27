import numpy as np
import random
import sys, time
from functions import production_function, wage_function, demand_function, innovate
from tradeutils import doAllTrades
from pricing import updatePricesAndConsume
from tqdm import tqdm

np.random.seed(1)

# Constants
case = '2x2'
iterations = 1000
n_countries = 2
countries = ['USA', 'China']
products= ['wine', 'cloth']
n_products = 2
alpha = np.array([[0.5, 0.5],
                  [0.5, 0.5]])  # output elasticity of labor

beta = np.array([[0.5, 0.5],
                 [0.5, 0.5]])  # output elasticity of capital

A = np.array([[0.5, 2.0],
              [0.2, 0.05]])  # Total Factor Productivity

shock = np.array([[0.5, 2.0],
                  [0.2, 0.8]])  # Total Factor Productivity

share = np.array([1,1])
# Number of citizens in each nation
citizens_per_nation = [100, 1000]


def gulden_vectorised(case, n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta, 
                      iterations=2000, Tr_time=500, trade_change=0.5, autarky_time= 10000, pricing_algorithm='cpmu', utility_algorithm='geometric', wage_algorithm = 'marginal_product', share=share,
                      csv = False, plot = False, shock = shock, shock_time = 10000, cm_time=10000, d=0.0,
                      innovation = False, innovation_time = 10000, gamma=1, eta = 0.01,weights=None, elasticities=None, sigma=None):
    
    ## Citizen-level
    # Initialize prices for each good in each nation
    prices = np.ones((n_countries, n_products))
    prices_vec = np.ones((n_countries, n_products, iterations))

    # Initialize industry choices for labor and capital for each citizen in each nation
    labor_choices = [np.zeros(citizens, dtype=int) for citizens in citizens_per_nation]
    capital_choices = [np.zeros((citizens, 2), dtype=int) for citizens in citizens_per_nation]    
    income = [np.zeros(citizens) for citizens in citizens_per_nation]
    
    # Allow capital choices to take both industry and nation if capital mobility is enabled
    for nation in range(n_countries):
        labor_choices[nation] = np.random.choice(range(n_products), citizens_per_nation[nation])
        capital_choices[nation][:,0] = np.random.choice(range(n_products), citizens_per_nation[nation])
        capital_choices[nation][:,1] = np.ones(citizens_per_nation[nation])*nation

    ## Nation-level
    # Calculate wages and returns to capital for each industry in each nation
    L = np.zeros((n_countries, n_products))
    K = np.zeros((n_countries, n_products))
    Q = np.zeros((n_countries, n_products))
    W = np.zeros((n_countries, n_products))
    R = np.zeros((n_countries, n_products))
    D = np.zeros((n_countries, n_products))
    S = np.zeros((n_countries, n_products))
    net_exports = np.zeros(S.shape, dtype=np.float128)

    tv = np.zeros((n_countries, n_countries, n_products))
    labor = np.zeros((n_countries, n_products,iterations))
    capital = np.zeros((n_countries, n_products,iterations))
    production = np.zeros((n_countries, n_products,iterations))
    demand = np.zeros((n_countries, n_products,iterations))
    supply = np.zeros((n_countries, n_products, iterations))
    net_exports_history = np.zeros((n_countries, n_products,iterations))
    wage = np.zeros((n_countries, n_products,iterations))
    returns = np.zeros((n_countries, n_products,iterations))
    utility = np.zeros((n_countries, iterations))
    gdp_vec = np.zeros((n_countries, iterations))
    gnp_vec = np.zeros((n_countries, iterations))

    try:
        if len(Tr_time)>1:
            T0 = Tr_time[0]
            T1 = Tr_time[1]
    except:
        T0 = Tr_time
        T1 = 0

    for t in tqdm(range(iterations)):

        Tr=False
        cap_mobility = False

        if t>=T0 or (T1>0 and t>=T1) :
            Tr=True

        if t>=autarky_time:
            if (T1>0 and t<T1):
                Tr=False

        if t>shock_time:
            A = shock

        if t>=cm_time:
            cap_mobility = True

        
        for nation in range(n_countries):
            # Initialise 
            should_change_labor = np.zeros(citizens_per_nation[nation])
            should_change_capital = np.zeros(citizens_per_nation[nation])

            # Citizen update
            # Re-assess labor and capital choices
            # Citizens decide whether to change their job or investment based on probability P
            max_wage_industry = np.argmax(W[nation,:])
            event_occurs = np.random.binomial(1, 0.004, size=citizens_per_nation[nation])
            should_change_labor = (event_occurs==1) * (W[nation, labor_choices[nation]]< W[nation, max_wage_industry]*(1+d))
            
            if cap_mobility:
                max_return_industry = np.unravel_index(np.argmax(R, axis=None), R.shape)
                should_change_capital = (event_occurs==1) * (R[capital_choices[nation][:,1], capital_choices[nation][:,0]]<R[max_return_industry]*(1+d))
                capital_choices[nation][should_change_capital==1] = (max_return_industry[1],max_return_industry[0])
            
            else:
                max_return_industry = np.argmax(R[nation,:])
                should_change_capital = (event_occurs==1) * (R[nation, capital_choices[nation][:,0]]< R[nation, max_return_industry]*(1+d))
                capital_choices[nation][should_change_capital==1] = (max_return_industry,nation)

            labor_choices[nation][should_change_labor==1] = max_wage_industry
            
            income[nation] = sum(W[nation, labor_choices[nation]] + R[capital_choices[nation][:,1], capital_choices[nation][:,0]])

            for industry in range(n_products):
                L[nation,industry] = 0 
                D[nation,industry] = income[nation]/(n_products*prices[nation,industry])
                demand[nation,industry,t] = D[nation,industry]
                L[nation,industry] = sum(labor_choices[nation] == industry)

                if cap_mobility:
                    K[nation, industry]=0
                    K[nation,industry] = sum((capital_choices[c][i,0]==industry) and (capital_choices[c][i,1]==nation) for c in range(n_countries) for i in range(citizens_per_nation[c]))
                
                else:
                    K[nation, industry]=0
                    K[nation,industry] = sum(capital_choices[nation][:,0] == industry)
                
                if K[nation, industry]<1:
                    K[nation, industry] = 1

            # Calculate labor, capital, production, wages, and returns for each industry in each nation
            # for industry in range(n_products):
                Q[nation,industry] = production_function(A[nation,industry], L[nation,industry], K[nation,industry],alpha[nation,industry], beta[nation,industry])
                W[nation, industry], R[nation, industry] =  wage_function(A[nation,industry], L[nation,industry], K[nation,industry], alpha[nation,industry], beta[nation,industry],p=prices[nation, industry],algorithm=wage_algorithm, share=share[nation])
                S[nation, industry] = Q[nation, industry]

                # Containers
                labor[nation,industry,t] = L[nation,industry]
                capital[nation,industry,t] = K[nation,industry]
                production[nation,industry,t] = Q[nation,industry]
                wage[nation,industry,t] = W[nation, industry]
                returns[nation,industry,t] = R[nation, industry]
            gnp_vec[nation,t] = sum(Q[nation,:])

        # Trade update
        if Tr:
            trades, net_exports = doAllTrades(tv, S, prices, trade_change)
            # net_exports = trades[1]
            S = Q- net_exports
        net_exports_history[:,:,t] = net_exports
            
        # Update prices and Consume
        prices, UT = updatePricesAndConsume(prices, D, S,pricing_algorithm, utility_algorithm, weights=weights, elasticities=elasticities, sigma=sigma)
        prices_vec[:,:,t] = prices
        utility[:,t] = UT
        gdp_vec[:,t] = income
        supply[:,:,t] = S
        
        # Innovate
        if (innovation) and (t>=innovation_time):
            A = innovate(A, K, gamma, eta)

                
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
                df.to_csv('csvs/vectorised_{}_{}.csv'.format(countries[nation], products[industry]))
    
    # Plot the results
    if plot:
        import matplotlib.pyplot as plt
        variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI', 'MRS']
        fig,ax = plt.subplots(2,5, figsize=(25, 10))
        for nation in range(n_countries):
            for industry in range(n_products):
                ax[0,0].plot(range(iterations), production[nation, industry, :], label='{}-{}'.format(countries[nation], products[industry]))
                ax[0,0].set_title('Production')
                # ax[0,0].axvline(x=Tr_time, ls= '--', color='k')
                # ax[0,0].text(Tr_time, np.max(production), 'Trade', ha='center', va='center',rotation='vertical', backgroundcolor='white')
                ax[0,0].axvline(x=shock_time, ls= '--', color='k')

                ax[1,0].plot(range(iterations), prices_vec[nation, industry, :], label='{}-{}'.format(countries[nation], products[industry]))
                ax[1,0].set_title('Prices')
                # ax[1,0].axvline(x=Tr_time, ls= '--', color='k')
                # ax[1,0].text(Tr_time, np.max(production), 'Trade', ha='center', va='center',rotation='vertical', backgroundcolor='white')
                ax[1,0].axvline(x=shock_time, ls= '--', color='k')

                ax[0,2].plot(range(iterations), demand[nation, industry, :], label='{}-{}'.format(countries[nation], products[industry]))
                ax[0,2].set_title('Demand')
                # ax[0,2].axvline(x=Tr_time, ls= '--', color='k' )
                ax[0,2].axvline(x=shock_time, ls= '--', color='k')

                ax[1,2].plot(range(iterations), supply[nation, industry, :], label='{}-{}'.format(countries[nation], products[industry]))
                ax[1,2].set_title('Supply')
                # ax[1,2].axvline(x=Tr_time, ls= '--', color='k' )
                ax[1,2].axvline(x=shock_time, ls= '--', color='k')

                ax[0,3].plot(range(iterations), labor[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]))
                ax[0,3].set_title('Labor')
                # ax[0,3].axvline(x=Tr_time, ls= '--', color='k' )
                ax[0,3].axvline(x=shock_time, ls= '--', color='k')

                ax[1,3].plot(range(iterations), capital[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]))
                ax[1,3].set_title('Capital')
                # ax[1,3].axvline(x=Tr_time, ls= '--', color='k' )
                ax[1,3].axvline(x=shock_time, ls= '--', color='k')

                ax[0,4].plot(range(iterations), wage[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]))
                ax[0,4].set_title('Wages')
                # ax[0,4].axvline(x=Tr_time, ls= '--', color='k' )
                ax[0,4].axvline(x=shock_time, ls= '--', color='k')

                ax[1,4].plot(range(iterations), returns[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]))
                ax[1,4].set_title('Returns')
                # ax[1,4].axvline(x=Tr_time, ls= '--', color='k' )
                ax[1,4].axvline(x=shock_time, ls= '--', color='k')

            ax[0,1].plot(range(iterations), gdp_vec[nation, :], label='{}-{}'.format(countries[nation], products[industry]))
            ax[0,1].set_title('GDP')
            # ax[0,1].axvline(x=Tr_time, ls= '--', color='k' )
            ax[0,1].axvline(x=shock_time, ls= '--', color='k')

            ax[1,1].plot(range(iterations), utility[nation, :], label='{}-{}'.format(countries[nation], products[industry]))
            ax[1,1].set_title('Utility')
            # ax[1,1].axvline(x=Tr_time, ls= '--', color='k' )
            ax[1,1].axvline(x=shock_time, ls= '--', color='k')

        handles, labels = ax[0,0].get_legend_handles_labels() 
        fig.legend(handles, labels, loc='lower center', ncol = n_countries, fontsize = 15)
        plt.savefig('plots/vectorised_{}.png'.format(case))
        
    return net_exports, income, production

# # model run
# t1 = time.time()
# gulden_vectorised('cap_mob', n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#                   iterations=2000, Tr_time=500, autarky_time=15000, pricing_algorithm='dgp', utility_algorithm='geometric',
#                   wage_algorithm='marginal_product', share=np.array([1,1]),
#                   shock = shock, shock_time=10000, cm_time=10000, plot=True, csv=False, d=0.0)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))

# t1 = time.time()
# gulden_vectorised('share_product', n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#                   iterations=2000, Tr_time=500, autarky_time=15000, pricing_algorithm='dgp', utility_algorithm='geometric',
#                   wage_algorithm='share_of_marginal_product', share=np.array([0.5,1]),
#                   shock = shock, shock_time=10000, cm_time=1000, plot=True, csv=False, d=0.0)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))






