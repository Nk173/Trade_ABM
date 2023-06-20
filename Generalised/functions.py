import random
import numpy as np
import sys
from typing import Dict, List

# Define the Cobb-Douglas production function
def production_function(A, alpha, L, beta, K):
    return A * (L ** alpha) * (K ** beta)

# Define the demand function
def demand_function(Y, P):
    
    D = np.zeros((len(P)))
    for p in range(len(P)):
        D[p] = Y / (len(P) * P[p])

    return D

# def compute_price_marginal_utilities(country):
#     from Agents import Nation
#     # set algorithm for utility function ('geometric' or 'ces')
#     algorithm = 'ces'
#     # parameter values for ces calculations
#     from init import weights, elasticities, sigma


#     consumption_hypothetical = country.supply.copy()
#     country.UT = country.utilityFunction(country.supply, algorithm=algorithm, weights=weights, elasticities=elasticities, sigma=sigma)
#     consumption_hypothetical['wine'] += 1
#     marginal_utility_wine = country.utilityFunction(consumption_hypothetical, algorithm=algorithm, weights=weights, elasticities=elasticities, sigma=sigma) - country.UT
#     consumption_hypothetical['wine'] -= 1
#     country.mrs["wine"] = 1
#     import math

#     # Prices (price of wine is set to 1 and is the reference good)
#     for i in range(1, len(country.industries)):
#         industry = country.industries[i]
#         consumption_hypothetical[industry] += 1
#         marginal_utility = country.utilityFunction(consumption_hypothetical, algorithm=algorithm, weights=weights, elasticities=elasticities, sigma=sigma) - country.UT
#         consumption_hypothetical[industry] -= 1
#         country.mrs[industry] = marginal_utility / marginal_utility_wine
#         error = abs(country.mrs[industry] - country.prices[industry])
#         if country.mrs[industry] >  country.prices[industry]:
#             country.prices[industry] = country.prices[industry] + (country.prices[industry] * min(0.02, 0.01 * error))
#         elif country.mrs[industry] < country.prices[industry]:
#             country.prices[industry] = country.prices[industry] - (country.prices[industry] * min(0.02, 0.01 * error))



# def compute_price_immediate_marginal_utility(country):

#     consumption_hypothetical = country.supply.copy()
#     country.UT = country.utilityFunction(country.supply)
#     consumption_hypothetical['wine'] += 1
#     marginal_utility_wine = country.utilityFunction(consumption_hypothetical) - country.UT
#     consumption_hypothetical['wine'] -= 1
#     country.mrs["wine"] = 1
#     import math

#     # Prices (price of wine is set to 1 and is the reference good)
#     for i in range(1, len(country.industries)):
#         industry = country.industries[i]
#         consumption_hypothetical[industry] += 1
#         marginal_utility = country.utilityFunction(consumption_hypothetical) - country.UT
#         consumption_hypothetical[industry] -= 1
#         country.mrs[industry] = marginal_utility / marginal_utility_wine
#         if (math.isfinite(country.mrs[industry])):
#             country.prices[industry] = country.mrs[industry]
#         else:
#             print("lame!")


# def resetAllMonetaryFlows(trade_volume: Dict[str, Dict[str, Dict[str, float]]],
#                           nationsdict: dict[str,Nation],
#                           industries: List[str], countries: List[str], nominal_good='wine'):
#     """
#     Reset all monetary flows in the trade_volume dictionary and calculate the trade cost
#     for every industry and every country to country pairing.

#     Args:
#     - trade_volume: a nested dictionary that represents the trade volume between countries for each industry. Negatives are exports...
#     - nationsdict: a dictionary that  links from names to nations objects
#     - industries: a list that contains the names of all industries.
#     - countries: a list that contains the names of all countries.

#     Returns:
#     None
#     """
#     ## you cannot trade more than what you make so we need to account for that
#     ## todo but there is a problem here: if you are exporting something to buy something else this can screw it up
#     # Set all elements of trade_volume['money'] to zero for every possible country
#     # for countryA in countries:
#     #     for countryB in countries:
#     #         trade_volume[nominal_good][countryA][countryB] = 0.0


#     ## we will compute this multiplier to help us reduce trade below total production, if needed
#     net_exports: Dict[str, dict[str, float]] = compute_net_export(countries,industries,trade_volume)
#     export_adjustment: Dict[str, dict[str, float]] = {}
#     nominal_budget_left: Dict[str, float] = {}
#     for country in countries:
#         export_adjustment[country] = {}
#         for industry in industries:
#             if industry != nominal_good:
#                 export_adjustment[country][industry] = 1 if net_exports[country][industry] >0 else min(1,nationsdict[country].production[industry]/abs(net_exports[country][industry]))
#                 if abs(export_adjustment[country][industry])>1:
#                     print(export_adjustment[country])
#                     print(nationsdict[country].production[industry])
#                     print(net_exports[country][industry])
#                     sys.exit(0)
#             else:
#                 nominal_budget_left[country] = nationsdict[country].production[nominal_good] - net_exports[country][nominal_good]

#     for countryA in countries:
#         for countryB in countries:
#             trade_volume[nominal_good][countryA][countryB] = 0.0


#     # Go through every industry and every country to country pairing
#     for industry in industries:
#         if industry == nominal_good:
#             continue
#         for countryA in countries:
#             for countryB in countries:
#                 if countryA == countryB:
#                     continue
#                 old_trade_volume = trade_volume[industry][countryA][countryB]
#                 # if country A is exporting to country B
#                 if old_trade_volume < 0.0:
#                     ## country B buys from country A at country A's values
#                     new_trade_volume = old_trade_volume * export_adjustment[countryA][industry]
#                     trade_cost = -trade_volume[industry][countryA][countryB] * nationsdict[countryA].prices[industry]
#                     ## check if we break through the remaining budget for importer
#                     if(nominal_budget_left[countryB]<abs(trade_cost)):
#                         trade_cost = nominal_budget_left[countryB]
#                         new_trade_volume = -trade_cost / nationsdict[countryA].prices[industry]

#                     if abs(trade_cost) > 1000 or abs(new_trade_volume) > 1000:
#                         print(trade_cost)
#                         print(new_trade_volume)
#                         print(industry)
#                         print(countryA)
#                         print(countryB)
#                         print(nationsdict[countryA].prices[industry])
#                         print(old_trade_volume)
#                         print(export_adjustment[countryA][industry])
#                         print(nominal_budget_left[countryB])
#                         sys.exit(0)

#                     trade_volume[industry][countryA][countryB] = new_trade_volume
#                     trade_volume[industry][countryB][countryA] = -new_trade_volume
#                     # record countryA importing money and countryB exporting it....
#                     trade_volume[nominal_good][countryA][countryB] += trade_cost
#                     trade_volume[nominal_good][countryB][countryA] -= trade_cost
#                     old_price = nationsdict[countryA].old_prices[industry]
#                     nominal_budget_left[countryA] += (trade_cost + old_trade_volume*old_price)
#                     nominal_budget_left[countryB] -= (trade_cost + old_trade_volume*old_price)




# def doAllTrades(trade_volume: Dict[str, Dict[str, Dict[str, float]]],industries: List[str],countries: List[str],
#                 nationsdict: dict[str,Nation],
#                 nominal_good: "wine"):
#     ## adjust due to price changes...
#     resetAllMonetaryFlows(trade_volume,
#                           nationsdict,
#                           industries,
#                           countries,
#                           nominal_good)

#     net_exports : Dict[str, Dict[str, float]] = compute_net_export(countries,industries,trade_volume)
#     ## find all possible trade
#     pairings: List[List[str]] = []  # create an empty list to store the pairings
#     for i in range(len(countries)): # iterate over the countries
#         for j in range(i + 1, len(countries)): # iterate over the remaining countries
#             for industry in industries:  # iterate over the industries
#                 if industry != nominal_good:
#                     pairings.append([countries[i], countries[j], industry]) # append the pairing to the list

#     random.shuffle(pairings) # shuffle the list of pairings
#     ## trade
#     for pairing in pairings:
#         doOneTrade(trade_volume,nationsdict,
#                    net_exports,
#                    pairing[2],
#                    pairing[1],
#                    pairing[0],
#                    nominal_good
#                    )
#     return {
#         "trade_volume" : trade_volume,
#         "net_exports": net_exports
#     }


# def compute_net_export(
#     countries: List[str],
#     industries: List[str],
#     trades: Dict[str, Dict[str, Dict[str, float]]],
# ) -> Dict[str, Dict[str, float]]:
#   """Computes net-export for every country and for every industry.

#   Args:
#     countries: A list of strings, the names of the countries.
#     industries: A list of strings, the names of the industries.
#     trades: A list of matrices of floats, a matrix for each industry.

#   Returns:
#     A dictionary mapping countries to a dictionary mapping industries to their net-exports.
#   """

#   # Initialize the net-exports dictionary.
#   net_exports: Dict[str, Dict[str, float]] = {}
#   for country in countries:
#     net_exports[country] = {}
#     for industry in industries:
#       net_exports[country][industry] = 0

#   # Iterate over the countries and industries.
#   for country in countries:
#     for industry in industries:
#       # Sum the exports from the country to all other countries.
#       for other_country in countries:
#         if country == other_country:
#           continue
#         net_exports[country][industry] -= trades[industry][country][other_country]

#   # Return the net-exports dictionary.
#   return net_exports


# def doOneTrade(trade_volume,
#                 nationsdict: dict[str,Nation],
#                net_exports: Dict[str, Dict[str, float]],
#                industry: str,
#                countryOne: str,
#                countryTwo: str,
#                nominal_good: str):

#     one = nationsdict[countryOne]
#     two = nationsdict[countryTwo]
#     ## if the prices are exactly the same (or have a very tiny difference), who cares?
#     difference = abs(one.prices[industry] - two.prices[industry])
#     percentage_difference = (difference / one.prices[industry]) * 100

#     if difference < 0.01 or percentage_difference < 5:
#         return
#     ## okay, now let's find who is going to export to who...

#     if  one.prices[industry] < two.prices[industry]:
#         exporterName = countryOne; exporter = one; importerName = countryTwo; importer = two;
#     else:
#         exporterName = countryTwo; exporter = two; importerName = countryOne; importer = one;

#     ##country one: how much do they hold of the item in the tradeable industry vs the nominal good (money)
#     inventory_exporter = exporter.production[industry] - net_exports[exporterName][industry]

#     ##country two: how much do they hold of the item in the tradeable industry vs the nominal good (money)
#     money_importer = importer.production[nominal_good] - net_exports[importerName][nominal_good]


#     ## let's assume we really just want to move 0.5 items from one country to another...
#     target_trade_change: float = 0.5
#     ## you buy at seller local price?
#     money_exchanged =  exporter.prices[industry]

#     if inventory_exporter > target_trade_change and money_importer > money_exchanged:
#         ## trade!
#         trade_volume[industry][exporterName][importerName] -= target_trade_change
#         trade_volume[industry][importerName][exporterName] += target_trade_change
#         ## money exchange
#         trade_volume[nominal_good][exporterName][importerName] += money_exchanged
#         trade_volume[nominal_good][importerName][exporterName] -= money_exchanged
#         ## you have to update this running thing as well...
#         net_exports[importerName][industry] -= target_trade_change
#         net_exports[exporterName][industry] += target_trade_change
#         net_exports[importerName][nominal_good] += money_exchanged
#         net_exports[exporterName][nominal_good] -= money_exchanged

