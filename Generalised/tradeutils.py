import random
from typing import Dict, List

from Agents import Nation


def resetAllMonetaryFlows(trade_volume: Dict[str, Dict[str, Dict[str, float]]],
                          nationsdict: dict[str,Nation],
                          industries: List[str], countries: List[str], nominal_good='wine'):
    """
    Reset all monetary flows in the trade_volume dictionary and calculate the trade cost
    for every industry and every country to country pairing.

    Args:
    - trade_volume: a nested dictionary that represents the trade volume between countries for each industry. Negatives are exports...
    - nationsdict: a dictionary that  links from names to nations objects
    - industries: a list that contains the names of all industries.
    - countries: a list that contains the names of all countries.

    Returns:
    None
    """
    # Set all elements of trade_volume['money'] to zero for every possible country
    for countryA in countries:
        for countryB in countries:
            trade_volume[nominal_good][countryA][countryB] = 0.0

    # Go through every industry and every country to country pairing
    for industry in industries:
        if industry == nominal_good:
            continue
        for countryA in countries:
            for countryB in countries:
                # if country A is exporting to country B
                if trade_volume[industry][countryA][countryB] < 0.0:
                    ## country B buys from country A at country A's values
                    trade_cost = -trade_volume[industry][countryA][countryB] * nationsdict[countryA].prices[industry]
                    # record countryA importing money and countryB exporting it....
                    trade_volume[nominal_good][countryA][countryB] += trade_cost
                    trade_volume[nominal_good][countryB][countryA] -= trade_cost


def doAllTrades(trade_volume: Dict[str, Dict[str, Dict[str, float]]],industries: List[str],countries: List[str],
                nationsdict: dict[str,Nation],
                nominal_good: "wine"):
    ## adjust due to price changes...
    resetAllMonetaryFlows(trade_volume,
                          nationsdict,
                          industries,
                          countries,
                          nominal_good)

    net_exports : Dict[str, Dict[str, float]] = compute_net_export(countries,industries,trade_volume)
    ## find all possible trade
    pairings: List[List[str]] = []  # create an empty list to store the pairings
    for i in range(len(countries)): # iterate over the countries
        for j in range(i + 1, len(countries)): # iterate over the remaining countries
            for industry in industries:  # iterate over the industries
                if industry != nominal_good:
                    pairings.append([countries[i], countries[j], industry]) # append the pairing to the list

    random.shuffle(pairings) # shuffle the list of pairings
    ## now for each pair we need to know:
    ## 1) how much they produce + trade in of good to trade
    ## 2) how much they produce + trade in of "money" (wine or whatever)
    ## 3) if they can trade...
    for pairing in pairings:
        doOneTrade(trade_volume,nationsdict,
                   net_exports,
                   pairing[2],
                   pairing[1],
                   pairing[0],
                   nominal_good
                   )
    return {
        "trade_volume" : trade_volume,
        "net_exports": net_exports
    }


def compute_net_export(
    countries: List[str],
    industries: List[str],
    trades: List[List[float]],
) -> Dict[str, Dict[str, float]]:
  """Computes net-export for every country and for every industry.

  Args:
    countries: A list of strings, the names of the countries.
    industries: A list of strings, the names of the industries.
    trades: A list of matrices of floats, a matrix for each industry.

  Returns:
    A dictionary mapping countries to a dictionary mapping industries to their net-exports.
  """

  # Initialize the net-exports dictionary.
  net_exports: Dict[str, Dict[str, float]] = {}
  for country in countries:
    net_exports[country] = {}
    for industry in industries:
      net_exports[country][industry] = 0

  # Iterate over the countries and industries.
  for country in countries:
    for industry in industries:
      # Sum the exports from the country to all other countries.
      for other_country in countries:
        if country == other_country:
          continue
        net_exports[country][industry] -= trades[industry][country][other_country]

  # Return the net-exports dictionary.
  return net_exports


def doOneTrade(trade_volume,
                nationsdict: dict[str,Nation],
               net_exports: Dict[str, Dict[str, float]],
               industry: str,
               countryOne: str,
               countryTwo: str,
               nominal_good: str):

    one = nationsdict[countryOne]
    two = nationsdict[countryTwo]
    ## if the prices are exactly the same (or have a very tiny difference), who cares?
    difference = abs(one.prices[industry] - two.prices[industry])
    percentage_difference = (difference / one.prices[industry]) * 100

    if difference < 0.01 or percentage_difference < 1:
        return
    ## okay, now let's find who is going to export to who...

    if  one.prices[industry]<two.prices[industry]:
        exporterName = countryOne; exporter = one; importerName = countryTwo; importer = two;
    else:
        exporterName = countryTwo; exporter = two; importerName = countryOne; importer = one;

    ##country one: how much do they hold of the item in the tradeable industry vs the nominal good (money)
    inventory_exporter = exporter.supply[industry] - net_exports[exporterName][industry]

    ##country two: how much do they hold of the item in the tradeable industry vs the nominal good (money)
    money_importer = importer.supply[nominal_good] - net_exports[importerName][nominal_good]


    ## let's assume we really just want to move 0.5 items from one country to another...
    target_trade_change: float = 0.5
    ## you buy at seller local price?
    money_exchanged =  exporter.prices[industry]

    if inventory_exporter > target_trade_change and money_importer > money_exchanged:
        ## trade!
        trade_volume[industry][exporterName][importerName] -= target_trade_change
        trade_volume[industry][importerName][exporterName] += target_trade_change
        ## money exchange
        trade_volume[nominal_good][exporterName][importerName] += money_exchanged
        trade_volume[nominal_good][importerName][exporterName] -= money_exchanged
        ## you have to update this running thing as well...
        net_exports[importerName][industry] -= target_trade_change
        net_exports[exporterName][industry] += target_trade_change
        net_exports[importerName][nominal_good] += money_exchanged
        net_exports[exporterName][nominal_good] -= money_exchanged
