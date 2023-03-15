import random
import numpy as np
from Agents import Nation, Citizen
import matplotlib.pyplot as plt

global countries, industries, development_shock

# 3 country 3 product case
countries = ['USA','CHINA','INDIA']
count = [100, 1000,500]
industries = ['wine', 'cloth', 'wheat']

P0 = [1,1,1]
P1 = [1,1,1]
P2 = [1,1,1]

A0 = [0.5,2, 0.25]
A1 = [0.2,0.05, 0.25]
A2 = [0.1,0.4, 0.25]

alpha0 = [0.5, 0.5, 0.5]
alpha1 = [0.5, 0.5, 0.5]

beta0 = [0.5, 0.5,0.5]
beta1 = [0.5, 0.5,0.5]

variables = ['production', 'demand','supply','labor','capital','wages', 'prices','ROI']
functions = ['get_production', 'get_demand', 'get_supply', 'get_labor', 'get_capital', 'get_wages', 'get_prices', 'get_ROI']


resultsU = {}
for c in countries:
    resultsU[c] = []
    
# Results Container
resultsdict = {}
for c in countries:
    resultsdict[c] = {}
    for v in variables:
        resultsdict[c][v]={}
        for i in industries:
            resultsdict[c][v][i] = []

# Utility Results Container
resultsU = {}
for c in countries:
    resultsU[c] = []

USA_ = Nation(countries[0], count[0], industries, countries, P0,  
              A0, alpha0, beta0)
CHINA_ = Nation(countries[1], count[1], industries,countries, P1,  
              A1, alpha1, beta1)
INDIA_ = Nation(countries[2], count[2], industries,countries, P2,  
              A2, alpha1, beta1)

# Nations Dictionary container
nationsdict={
    "USA":USA_,
    "CHINA":CHINA_,
    "INDIA":INDIA_,
}

            
# Trade Volume Dictionary
tv={}
for c in countries:
    tv[c]={}
    for n in countries:
        tv[c][n]=0

# Time Evolution of the Model
for t in range(1000):  
    # print('step',t)
    tr = False
    partner_develops = False
    if t>500:
        tr=True

    for c in countries:
        nationsdict[c].update(trade_volume = tv, trade=tr, nationsdict=nationsdict, capital_mobility = False, partner_develops = partner_develops)
        if t>=500:    
            tv = nationsdict[c].get_trade_volume()
            # print(tv)
        for v in range(len(variables)):
            for i in industries:
                resultsdict[c][variables[v]][i].append(getattr(nationsdict[c],functions[v])()[i])
        
        resultsU[c].append(nationsdict[c].get_utility())

## Plotting results
fig, axs = plt.subplots(5,2, figsize=(15, 20), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.1)
axs = axs.ravel()

for c in countries:
    for k in range(len(variables)):
        for i in industries:
            axs[k].plot(resultsdict[c][variables[k]][i], label='{}-{}-{}'.format(c, variables[k], i))
            axs[k].legend()
    for k in range(len(variables), len(variables)+1):
        axs[k].plot(resultsU[c], label='{}-{}'.format(c,'utility'))
        axs[k].legend()
        
plt.suptitle('3 countries, 3 industries with A0={},\n A1={}, \n A2={}'.format(A0,A1,A2), fontsize=16)
fig.savefig('plots/3C3I_3.png')
