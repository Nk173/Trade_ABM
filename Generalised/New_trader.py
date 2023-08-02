# from init import case, countries, count, industries, P, A, alpha, beta, shock
from pricing import compute_price_immediate_marginal_utility, compute_price_marginal_utilities
from pricing import gd_pricing, demand_gap_pricing
from wages import wagesAsShareOfMarginalProduct, wageAsMarginalProductROIAsResidual, wageAsShareOfProduct

case = 'new_trader'
countries = ['USA','CHINA','INDIA']
count = [100, 1000, 1000]
industries = ['wine','cloth']

P={}
P['USA'] =   [1,1]
P['CHINA'] = [1,1]
P['INDIA'] = [1,1]

A={}
A['USA']=    [0.5, 2]
A['CHINA'] = [0.2, 0.05]
A['INDIA'] = [0.2, 0.05]

alpha={}
alpha['USA'] =   [0.5, 0.5]
alpha['CHINA'] = [0.5, 0.5]
alpha['INDIA'] = [0.5, 0.5]

beta={}
beta['USA'] =   [0.5, 0.5]
beta['CHINA'] = [0.5, 0.5]
beta['INDIA'] = [0.5, 0.5]   

# 3-product case
# P={}
# P['USA'] =   [1,1,1]
# P['CHINA'] = [1,1,1]
# P['INDIA'] = [1,1,1]

# A={}
# A['USA']=    [0.5, 2, 1]
# A['CHINA'] = [0.2, 0.05, 1]
# A['INDIA'] = [0.15, 0.1, 1]

# alpha={}
# alpha['USA'] =   [0.5, 0.5, 0.5]
# alpha['CHINA'] = [0.5, 0.5, 0.5]
# alpha['INDIA'] = [0.5, 0.5, 0.5]

# beta={}
# beta['USA'] =   [0.5, 0.5,0.5]
# beta['CHINA'] = [0.5, 0.5,0.5]
# beta['INDIA'] = [0.5, 0.5,0.5]   

shock = [0.2, 0.8]

def gulden(case=case, countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta,
        total_time = 3000, trade_time = 4000, trading_countries = countries,
        pd_time=10000, shock=shock, cm_time=6000, autarky_time=5000, new_trader_time=2000,
        pricing_algorithm =compute_price_marginal_utilities, wage_algorithm = wageAsMarginalProductROIAsResidual, utility_algorithm = 'geometric', plot=True, d=0):
    
    from typing import Dict
    from functions import production_function, demand_function
    from tradeutils import doAllTrades
    from Agents import Nation
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm 

    ## Set seed
    random.seed(0)

    ## Variables of Interest to Output from the model
    variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI', 'MRS']
    functions = ['get_production', 'get_demand', 'get_traded', 'get_labor', 'get_capital', 'get_wages', 'get_prices', 'get_ROI','get_MRS']

    if pricing_algorithm==demand_gap_pricing:
        variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI']
        functions = ['get_production', 'get_demand', 'get_traded', 'get_labor', 'get_capital', 'get_wages', 'get_prices', 'get_ROI']

    ## 
    inst = {}
    nationsdict={}
    for c in range(len(countries)):
        inst[c] = Nation(countries[c], count[c], industries, countries, P[countries[c]],  
                A[countries[c]], alpha[countries[c]], beta[countries[c]],
                        pricing_algorithm=pricing_algorithm,
                        utility_algorithm=utility_algorithm,
                        wage_algorithm = wage_algorithm,d=d)
        nationsdict[countries[c]]=inst[c]

    # Results Container
    resultsdict = {}
    for c in countries:
        resultsdict[c] = {}
        for v in variables:
            resultsdict[c][v]={}
            for i in industries:
                resultsdict[c][v][i] = []

    # Trade Volume Dictionary
    tv: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i in industries:
        tv[i] = {}
        for c in countries:
            tv[i][c]={}
            for n in countries:
                tv[i][c][n]=0.0


    # Utility Results Container
    resultsU = {}
    for c in countries:
        resultsU[c] = []

    try:
        if len(trade_time)>1:
            T0 = trade_time[0]
            T1 = trade_time[1]
    except:
        T0 = trade_time
        T1 = 0

# Time Evolution of the Model
    for t in tqdm(range(total_time)):
        # Markers
        tr = False
        partner_develops = False
        capital_mobility = False

        if t>=T0 or (T1>0 and t>=T1) :
            tr=True

        if t>=autarky_time:
            if (T1>0 and t<T1):
                tr=False

        if t==pd_time:
            partner_develops=True
            nationsdict[countries[1]].A = shock

        if t>=cm_time:
            capital_mobility=True

        if t==new_trader_time:
            trading_countries = countries
            
        for c in countries:
            nationsdict[c].update(nationsdict=nationsdict, capital_mobility=capital_mobility)

        if tr:
            trades = doAllTrades(tv, industries, trading_countries, nationsdict, "wine")
            #tv = trades["trade_volume"] not needed, this is exported anyway...
            net_exports=trades["net_exports"]
            # print("t is " + str(t))
            # print(tv["wheat"]["USA"]["INDIA"])

        ## update trading....
        try: 
            for c in countries:
                nationsdict[c].updatePricesAndConsume(trade=tr,country_export=net_exports[c] if tr else None)

        except:
            for c in trading_countries:
                nationsdict[c].updatePricesAndConsume(trade=tr,country_export=net_exports[c] if tr else None)

            nationsdict['INDIA'].updatePricesAndConsume(trade=False, country_export=None)

        for c in countries:
            for v in range(len(variables)):
                for i in industries:
                        resultsdict[c][variables[v]][i].append(getattr(nationsdict[c],functions[v])()[i])
                    
            resultsU[c].append(nationsdict[c].get_utility())        
            
            

    production = {}
    for c in countries:
        production[c] = nationsdict[c].production

    print(production)

    ## Plotting results
    if plot == True:
        fig, axs = plt.subplots(5,2, figsize=(30, 30), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.1)
        axs = axs.ravel()

        
        for c in countries:
            for k in range(len(variables)):
                for i in industries:
                    axs[k].plot(resultsdict[c][variables[k]][i], label='{}-{}-{}'.format(c, variables[k], i))
                    axs[k].set_title('{}'.format(variables[k]))
                    axs[k].legend(prop={'size': 10})
                    
            for k in range(len(variables), len(variables)+1):
                axs[k].plot(resultsU[c], label='{}-{}'.format(c,'utility'))
                axs[k].legend(prop={'size': 10})
                axs[k].set_title('{}'.format('Utility'))
        
        # axs[k].plot(sum(resultsU.values()), label='{}-{}'.format('World','utility'))
        plt.suptitle('Generalised {}'.format(case), fontsize=16)
        plt.savefig('plots/{}_dgp.png'.format(case))
    return production, resultsdict

# Model--Run
production, resultsdict = gulden(case='new_trader_2', countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta, 
                                 total_time = 2000, trade_time = 500, trading_countries=['USA','CHINA'],
                                 pd_time=10000, shock=shock, cm_time=6000, autarky_time=5000, new_trader_time=1000,
                                 pricing_algorithm =demand_gap_pricing, wage_algorithm = wageAsMarginalProductROIAsResidual, utility_algorithm = 'geometric', d=0.000000005)

production, resultsdict = gulden(case='3-traders_2', countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta, 
                                 total_time = 2000, trade_time = 500, trading_countries=['USA','CHINA','INDIA'],
                                 pd_time=10000, shock=shock, cm_time=6000, autarky_time=5000,
                                 pricing_algorithm =demand_gap_pricing, wage_algorithm = wageAsMarginalProductROIAsResidual, utility_algorithm = 'geometric', d=0.000000005)


# Model run with different pricing algorithm


