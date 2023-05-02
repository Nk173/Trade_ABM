
def compute_price_marginal_utilities(country):

    consumption_hypothetical = country.supply.copy()
    country.UT = country.utilityFunction(country.supply)
    consumption_hypothetical['wine'] += 1
    marginal_utility_wine = country.utilityFunction(consumption_hypothetical) - country.UT
    consumption_hypothetical['wine'] -= 1
    country.mrs["wine"] = 1
    import math

    # Prices (price of wine is set to 1 and is the reference good)
    for i in range(1, len(country.industries)):
        industry = country.industries[i]
        consumption_hypothetical[industry] += 1
        marginal_utility = country.utilityFunction(consumption_hypothetical) - country.UT
        consumption_hypothetical[industry] -= 1
        country.mrs[industry] = marginal_utility / marginal_utility_wine
        error = abs(country.mrs[industry] - country.prices[industry])
        if country.mrs[industry] >  country.prices[industry]:
            country.prices[industry] = country.prices[industry] + (country.prices[industry] * min(0.002, 0.001 * error))
        elif country.mrs[industry] < country.prices[industry]:
            country.prices[industry] = country.prices[industry] - (country.prices[industry] * min(0.002, 0.001 * error))



def compute_price_immediate_marginal_utility(country):

    consumption_hypothetical = country.supply.copy()
    country.UT = country.utilityFunction(country.supply)
    consumption_hypothetical['wine'] += 1
    marginal_utility_wine = country.utilityFunction(consumption_hypothetical) - country.UT
    consumption_hypothetical['wine'] -= 1
    country.mrs["wine"] = 1
    import math

    # Prices (price of wine is set to 1 and is the reference good)
    for i in range(1, len(country.industries)):
        industry = country.industries[i]
        consumption_hypothetical[industry] += 1
        marginal_utility = country.utilityFunction(consumption_hypothetical) - country.UT
        consumption_hypothetical[industry] -= 1
        country.mrs[industry] = marginal_utility / marginal_utility_wine
        if (math.isfinite(country.mrs[industry])):
            country.prices[industry] = country.mrs[industry]
        else:
            print("lame!")


