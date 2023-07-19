import numpy as np

def updatePricesAndConsume(prices, D, S ,pricing, utility='geometric'):
    consumption_hypothetical = S.copy()
    UT = utilityFunction(consumption_hypothetical, algorithm=utility)
    if pricing == 'dgp':
        prices[D>S] = prices[D>S] + prices[D>S]*.02
        prices[D<S] = prices[D<S] - prices[D<S]*.02

    elif pricing == 'cpmu':
        # consumption_hypothetical = S.copy()
        # UT = utilityFunction(consumption_hypothetical, algorithm=utility)
        mrs = np.ones((S.shape))
        consumption_hypothetical[:,0] += 1
        marginal_utility_wine= utilityFunction(consumption_hypothetical, algorithm=utility) - UT
        consumption_hypothetical[:,0] -= 1

        for i in range(1, S.shape[1]):
            consumption_hypothetical[:,i] += 1
            marginal_utility = utilityFunction(consumption_hypothetical, algorithm=utility) - UT
            consumption_hypothetical[:,i] -= 1
            mrs[:,i] = marginal_utility / marginal_utility_wine

        error = abs(mrs- prices)
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if mrs[i,j] > prices[i,j]:
                    prices[i,j] = prices[i,j] + prices[i,j]*min(0.02, error[i,j])
                elif mrs[i,j] < prices[i,j]:
                    prices[i,j] = prices[i,j] - prices[i,j]*min(0.02, error[i,j])
        # # positive = (mrs > prices)
        # # negative = (mrs < prices)
        # # if len(error[positive])==0 and len(error[negative])==0:
        # #     print('error')
        # prices[(positive)] = prices[(positive)] + prices[(positive)]*min(0.02, error[positive])
        # prices[(negative)] = prices[(negative)] - prices[(negative)]*min(0.02, error[negative])
    prices[:,0] = 1
    return prices, UT    
    
# vectorise the above funciton
# def updatePricesAndConsume(prices, D, S ,pricing):
#    if pricing == 'dgp':
#       prices[D>S] = prices[D>S] + prices[D>S]*.02
#      prices[D<S] = prices[D<S] - prices[D<S]*.02
# 
#   elif pricing == 'cpmu':
#     consumption_hypothetical = S.copy()
#    UT = utilityFunction(consumption_hypothetical, algorithm='geometric')
#  mrs = np.ones((S.shape))
# for i in range(1, S.shape[1]):
#   consumption_hypothetical[:,i] += 1
# marginal_utility = utilityFunction(consumption_hypothetical, algorithm='geometric') - UT
# consumption_hypothetical[:,i] -= 1
# mrs[:,i] = marginal_utility / marginal_utility_wine
# 
# error = abs(mrs- prices)
# positive = (mrs > prices)
# negative = (mrs < prices)
# if len(error[positive])==0 and len(error[negative])==0:
#     print('error')
# prices[(positive)] = prices[(positive)] + prices[(positive)]*min(0.02, error[positive])
# prices[(negative)] = prices[(negative)] - prices[(negative)]*min(0.02, error[negative])
# prices[:,0] = 1
# return prices


#     

def utilityFunction(consumption, algorithm, weights=None, elasticities=None, sigma=None):
    UT = 1
    if algorithm == 'geometric':
        for i in range(consumption.shape[1]):
            UT = UT * consumption[:,i]
        UT = UT**(1/(consumption.shape[1]))

    if algorithm == 'ces':
        w = weights
        s = elasticities
        p = sigma
        weighted_consumption = np.zeros((consumption.shape[1],1))
        for i in range(consumption.shape[1]):
            weighted_consumption[i] = w[i] * (consumption[:,i]**s[i])
        UT = np.sum(weighted_consumption) ** (1 / p)

    return UT