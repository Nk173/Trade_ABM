import numpy as np

def updatePricesAndConsume(prices, D, S, tariffs, pricing, utility, weights, elasticities, sigma):
    # Initialize consumption and utility
    consumption_hypothetical = S.copy()
    UT = utilityFunction(consumption_hypothetical, weights, elasticities, sigma, algorithm=utility)

    if pricing == 'dgp':
        # Adjust prices using demand-supply dynamics and effective prices
        for nation in range(prices.shape[0]):
            for industry in range(prices.shape[1]):
                if D[nation, industry] > S[nation, industry]:
                    prices[nation, industry] = prices[nation, industry] + prices[nation, industry] * .02
                elif D[nation, industry] < S[nation, industry]:
                    prices[nation, industry] = prices[nation, industry] - prices[nation, industry]*.02
                prices[nation, industry] = max(prices[nation, industry],0.0001)

    elif pricing == 'cpmu':
        mrs = np.ones(S.shape)
        consumption_hypothetical[:, 0] += 1
        marginal_utility_wine = (
            utilityFunction(consumption_hypothetical, weights, elasticities, sigma, algorithm=utility) - UT
        )
        consumption_hypothetical[:, 0] -= 1

        for i in range(1, S.shape[1]):
            consumption_hypothetical[:, i] += 1
            marginal_utility = (
                utilityFunction(consumption_hypothetical, weights, elasticities, sigma, algorithm=utility) - UT
            )
            consumption_hypothetical[:, i] -= 1
            mrs[:, i] = marginal_utility / marginal_utility_wine

        # Adjust prices using marginal rates of substitution and tariffs
        for nation in range(S.shape[0]):
            for industry in range(S.shape[1]):
                error = abs(mrs[nation, industry] - prices[nation, industry])
                if mrs[nation, industry] > prices[nation, industry]:
                    prices[nation, industry] += prices[nation, industry] * min(0.02, error+0.0001)
                elif mrs[nation, industry] < prices[nation, industry]:
                    prices[nation, industry] -= prices[nation, industry] * min(0.02, error+0.0001)

    prices[:, 0] = 1  # Anchor first product price
    return prices, UT


def utilityFunction(consumption,  weights, elasticities, sigma, algorithm, epsilon=1e-4):
    UT = 1
    if algorithm == 'geometric':
        # Ensure all values are positive and not zero
        adjusted_consumption = consumption + epsilon
        UT = np.prod(adjusted_consumption, axis=1)**(1/consumption.shape[1])

        # for i in range(consumption.shape[1]):
        #     UT = UT * consumption[:,i]
        # UT = UT**(1/(consumption.shape[1]))

    if algorithm == 'ces':
        w = weights
        s = elasticities
        p = sigma
        weighted_consumption = 0
        for i in range(consumption.shape[1]):
            weighted_consumption += w[i] * (consumption[:,i]**s[i])
        UT = (weighted_consumption) ** (1 / p)

    return UT