## Setup Gomory Baumol run with autarky: 0-500,  Trade:500-1000, partner_develops:1000-1500, autarky: 1500:2000, and trade: 2000-3000
from init import case, countries, count, industries, P, A, alpha, beta, shock
from typing import Dict
from pricing import compute_price_immediate_marginal_utility, compute_price_marginal_utilities
from tradeutils import doAllTrades
from functions import production_function, demand_function
from Agents import Nation
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import pandas as pd

random.seed(0)

## 
variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI', 'MRS']
functions = ['get_production', 'get_demand', 'get_traded', 'get_labor', 'get_capital', 'get_wages', 'get_prices', 'get_ROI','get_MRS']


## 
inst = {}
nationsdict={}
for c in range(len(countries)):
    inst[c] = Nation(countries[c], count[c], industries, countries, P[countries[c]],  
                    A[countries[c]], alpha[countries[c]], beta[countries[c]],
                    pricing_algorithm=compute_price_marginal_utilities)
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


# Time Evolution of the Model
for t in tqdm(range(5000)):
    # print('step',t)
    tr = False
    partner_develops = False
    if (t>=500 and t<1500) or t>2000:
        tr=True
    else:
        tr=False

    if t==1000:
        if partner_develops:
            for c in countries:
                inst[c] = Nation(countries[c], count[c], industries, countries, P[countries[c]],  
                                shock[countries[c]], alpha[countries[c]], beta[countries[c]],
                                pricing_algorithm=compute_price_marginal_utilities)
                nationsdict[countries[c]]=inst[c]
    

    for c in countries:
        nationsdict[c].update(nationsdict=nationsdict)
    if tr:
        trades = doAllTrades(tv, industries, countries, nationsdict, "wine")
        #tv = trades["trade_volume"] not needed, this is exported anyway...
        net_exports=trades["net_exports"]
            # print("t is " + str(t))
            # print(tv["wheat"]["USA"]["INDIA"])

        ## update trading....
    for c in countries:
        nationsdict[c].updatePricesAndConsume(trade=tr,country_export=net_exports[c] if tr else None)

        for v in range(len(variables)):
            for i in industries:
                resultsdict[c][variables[v]][i].append(getattr(nationsdict[c],functions[v])()[i])
        resultsU[c].append(nationsdict[c].get_utility())

exports = {}
for c in countries:
    exports[c] = nationsdict[c].production
    # print(nationsdict[c].production)

## Plotting results
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

plt.suptitle('Generalised {}'.format(case), fontsize=16)
plt.savefig('Generalised/plots/{}_gb.png'.format(case))