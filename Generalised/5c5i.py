## Model with Evolving R: R = A*(1/p)
import random
import numpy as np
from Agents import Nation, Citizen
import matplotlib.pyplot as plt

global countries, industries, development_shock

# 5 country 5 product case
countries = ['USA','JAPAN','CHINA','INDIA','GHANA']
count = [200, 200, 200, 200, 200]
count = [200, 200, 200, 200, 200]
industries = ['wine', 'cloth', 'wheat', 'computers','coal']

P0 = [1,1,1,1,1]
P1 = [1,1,1,1,1]
P2 = [1,1,1,1,1]
P3 = [1,1,1,1,1]
P4 = [1,1,1,1,1]

A0 = [1, 0.5, 0.5, 0.5, 0.5]
A1 = [0.5, 1, 0.5, 0.5, 0.5]
A2 = [0.5, 0.5, 1, 0.5, 0.5]
A3 = [0.5, 0.5, 0.5, 1, 0.5]
A4 = [0.5, 0.5, 0.5, 0.5, 1]

alpha0 = [0.5, 0.5, 0.5, 0.5, 0.5]
alpha1 = [0.5, 0.5, 0.5, 0.5, 0.5]
alpha2 = [0.5, 0.5, 0.5, 0.5, 0.5]
alpha3 = [0.5, 0.5, 0.5, 0.5, 0.5]
alpha4 = [0.5, 0.5, 0.5, 0.5, 0.5]

beta0 = [0.5, 0.5,0.5, 0.5,0.5]
beta1 = [0.5, 0.5,0.5,0.5,0.5]
beta2 = [0.5, 0.5,0.5,0.5,0.5]
beta3 = [0.5, 0.5,0.5,0.5,0.5]
beta4 = [0.5, 0.5,0.5,0.5,0.5]

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
JAPAN_ = Nation(countries[1], count[1], industries,countries, P1,  
              A1, alpha1, beta1)
CHINA_ = Nation(countries[2], count[2], industries,countries, P2,  
              A2, alpha1, beta1)
INDIA_ = Nation(countries[3], count[3], industries,countries, P3,  
              A3, alpha1, beta1)
GHANA_ = Nation(countries[4], count[4], industries,countries, P4,  
              A4, alpha1, beta1)

# Nations Dictionary container
nationsdict={
    "USA":USA_,
    "JAPAN":JAPAN_,
    "CHINA":CHINA_,
    "INDIA":INDIA_,
    "GHANA":GHANA_
}

            
# Trade Volume Dictionary
tv={}
for c in countries:
    tv[c]={}
    for n in countries:
        tv[c][n]=0

# Time Evolution of the Model
for t in range(2000):  
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
fig, axs = plt.subplots(5,2, figsize=(30, 30), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.1)
axs = axs.ravel()

for c in countries:
    for k in range(len(variables)):
        for i in industries:
            axs[k].plot(np.log(resultsdict[c][variables[k]][i]), label='{}-{}-{}'.format(c, variables[k], i))
            axs[k].legend(prop={'size': 6})
    for k in range(len(variables), len(variables)+1):
        axs[k].plot(resultsU[c], label='{}-{}'.format(c,'utility'))
        axs[k].legend(prop={'size': 6})
        
plt.suptitle('5 countries, 5 industries with Uniform'.format(A0,A1,A2,A3,A4), fontsize=16)
fig.savefig('plots/5C5I_uniform.png')
