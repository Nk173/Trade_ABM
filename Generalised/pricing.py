import numpy as np
import random
from functions import demand_function

def compute_price_marginal_utilities(country, algorithm):
    
    # set algorithm for utility function ('geometric' or 'ces')
    
    # parameter values for ces calculations
    from init import weights, elasticities, sigma
    consumption_hypothetical = country.supply.copy()
    country.UT = country.utilityFunction(country.supply, algorithm=algorithm, weights=weights, elasticities=elasticities, sigma=sigma)
    consumption_hypothetical['wine'] += 1
    marginal_utility_wine = country.utilityFunction(consumption_hypothetical, algorithm=algorithm, weights=weights, elasticities=elasticities, sigma=sigma) - country.UT
    consumption_hypothetical['wine'] -= 1
    country.mrs["wine"] = 1
    import math

    # Prices (price of wine is set to 1 and is the reference good)
    for i in range(1, len(country.industries)):
        industry = country.industries[i]
        consumption_hypothetical[industry] += 1
        marginal_utility = country.utilityFunction(consumption_hypothetical, algorithm=algorithm, weights=weights, elasticities=elasticities, sigma=sigma) - country.UT
        consumption_hypothetical[industry] -= 1
        country.mrs[industry] = marginal_utility / marginal_utility_wine
        error = abs(country.mrs[industry] - country.prices[industry])
        if country.mrs[industry] >  country.prices[industry]:
            country.prices[industry] = country.prices[industry] + (country.prices[industry] * min(0.02, 0.01 * error))
        elif country.mrs[industry] < country.prices[industry]:
            country.prices[industry] = country.prices[industry] - (country.prices[industry] * min(0.02, 0.01 * error))

def demand_gap_pricing(country):
    # Prices (price of wine is set to 1 and is the reference good)
    industries = country.industries
    for i in range(1, len(industries)):
        if country.demand[industries[i]] > country.supply[industries[i]]:
            country.prices[industries[i]] = country.prices[industries[i]] + (country.prices[industries[i]] * 0.02)
        else:
            country.prices[industries[i]] = country.prices[industries[i]] - (country.prices[industries[i]] * 0.02)


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


import numpy as np

def bandit_gradient_descent_learn(previous_loss: float,
                                  current_loss:float,
                                  previous_x,
                                  current_x):
    numerator = (current_loss - previous_loss)
    gradients = np.zeros(len(previous_x))

    if abs(numerator)<=0.0001:
        return gradients

    for j in range(len(previous_x)):
        epsilon = current_x[j] -  previous_x[j]
        if(abs(epsilon)<.00001):
            epsilon = (random.random()-0.5)*0.001
        # Finite difference method
        gradients[j] = (current_loss - previous_loss) / epsilon

    return gradients

def bandit_gradient_descent_move(learning_rate: float,
                                  estimated_gradients,
                                  previous_x):
    change = learning_rate * estimated_gradients
    change[change>0.1] = 0.1
    change[change<-0.1] = -0.1
    change[change==0] = random.random()*.2 - 0.1
    return previous_x - change


def gd_pricing(country):
    industries = country.industries

    if sum(country.old_demand.values()) == 0:
        return
    ## set up  learning rate
    learning_rate = 0.01
    # add up the loss...
    loss = 0
    for k in range(0,10):
        loss = 0
        demand = country.compute_hypothetical_demand(list(country.prices.values()),industries)
        for i in range(0, len(industries)):
            loss += abs( demand[industries[i]] - country.supply[industries[i]])
        prices_but_wine = [country.prices[industry] for industry in industries]
        prices_but_wine = np.array(prices_but_wine[1:])
        print(loss)

        if not ("estimated_gradients" in country.other_variables):
            country.other_variables["estimated_gradients"] = sample_spherical(len(industries)-1)*2
        else:
            country.other_variables["estimated_gradients"] = \
                bandit_gradient_descent_learn(country.other_variables["previous_loss"],
                                              current_loss=loss,
                                              previous_x=country.other_variables["previous_x"],
                                              current_x= prices_but_wine)

        next_x = bandit_gradient_descent_move(learning_rate,
                                              country.other_variables["estimated_gradients"],
                                              prices_but_wine)
        next_x[next_x<0.1] = 0.1
        country.other_variables["previous_loss"] = loss
        country.other_variables["previous_x"] = prices_but_wine
        country.prices = dict(zip(industries, [1] + list(next_x)))
        learning_rate = learning_rate * .98


