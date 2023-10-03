import numpy as np
import random
import sys

def compute_net_export(S, trades):
    # Create a new matrix 'net_exports' with the same shape as 'S' and populate it with zeros.
    net_exports = np.zeros(S.shape, dtype=np.float128)
    # Create a boolean mask of the same shape as the first dimension of 'S' (number of countries) and populate it with True.
    mask = np.ones(S.shape[0], dtype=bool)
    # Loop over each nation
    for i in range(S.shape[0]):
        # Set the current nation's mask value to False,
        # this means when we use this mask, we'll be looking at trades with all other nations EXCEPT the current nation.
        mask[i] = False
        # Calculate the negative sum of all trades that the current nation (i) does with other nations.
        # We do this by indexing the 'trades' tensor with the current nation and the mask.
        # This gives us a net export (negative trade) value for each industry for the current nation.
        net_exports[i] = -np.sum(trades[i, mask], axis=0)
        # Reset the current nation's mask value to True so it's ready for the next iteration.
        mask[i] = True

        # Ensure net exports don't exceed the supply of the product for the current nation.
        # If for some reason, the net exports calculated exceed the supply,
        # they are clipped to the maximum supply value of the nation.
        net_exports[i] = np.clip(net_exports[i], None, S[i])
    return net_exports

# def compute_net_export(S, trades):
#     total_imports = np.sum(trades, axis=0)
#     net_exports = -total_imports + trades
#     return net_exports



def resetAllMonetaryFlows(trade_volume, S, prices):
    # Compute net exports for all countries and industries.
    net_exports = compute_net_export(S, trade_volume)

    # Initialize arrays to adjust exports based on net exports and to track the remaining budget for each country.
    export_adjustment = np.zeros((S.shape), dtype=np.float128)
    nominal_budget_left = np.zeros((S.shape[0]), dtype=np.float128)

    for i in range(S.shape[0]):
        # Loop through each country.
        for j in range(S.shape[1]):
            # Loop through each industry.
            if j > 0:
                # Ignore the 0th industry.
                np.seterr(divide='ignore', invalid='ignore')

                # If a country has positive net exports for an industry, set the adjustment to 1. Otherwise, determine the adjustment based on supply and net exports.
                export_adjustment[i, j] = 1 if net_exports[i, j] >0 else min(1, S[i, j]/abs(net_exports[i,j]) + 0.001)

                # Error check to ensure the adjustment is not more than 1.
                if abs(export_adjustment[i,j])>1:
                    print('Error1')
                    sys.exit(0) 
                else:
                    # Calculate the remaining budget for the country.
                    nominal_budget_left[i] = S[i, 0] - net_exports[i,0]

    # Reset the total trade volume for each country pair.
    for c1 in range(S.shape[0]):
        for c2 in range(S.shape[0]):
            trade_volume[c1,c2,0] = 0.0
            trade_volume = trade_volume.astype(np.float128)

    # Readjust trade volumes based on the export adjustment and the budget of trading countries.
    for i in range(1,S.shape[1]):
        for c1 in range(S.shape[0]):
            for c2 in range(S.shape[0]):
                if c1==c2:
                    # Skip the case where the two countries are the same.
                    continue
                # Calculate new trade volume based on the old volume and the export adjustment.
                old_trade_volume = trade_volume[c1,c2,i]

                # Adjust the trade cost if it exceeds the budget of the importing country.
                if old_trade_volume < 0.0:
                    new_trade_volume = old_trade_volume * export_adjustment[c1,i]
                    trade_cost = -trade_volume[c1,c2,i]*prices[c1,i]

                    if nominal_budget_left[c2] < abs(trade_cost):
                        trade_cost = nominal_budget_left[c2]
                        new_trade_volume = -trade_cost/prices[c1,i]

                    # Update trade volumes and costs between the two countries.
                    trade_volume[c1,c2,i] = new_trade_volume
                    trade_volume[c2,c1,i] = -new_trade_volume
                    trade_volume[c1,c2,0] += trade_cost
                    trade_volume[c2,c1,0] -= trade_cost

                    # Update prices and remaining budget based on the recalibrated trade flows.
                    old_price = prices[c1,i]
                    nominal_budget_left[c1] += (trade_cost + old_trade_volume * old_price)
                    nominal_budget_left[c2] -= (trade_cost + old_trade_volume * old_price)
    return trade_volume


def doAllTrades(trade_volume, S, prices, trade_chage):  
        trade_volume = resetAllMonetaryFlows(trade_volume, S, prices)
        net_exports = compute_net_export(S, trade_volume)
        # pairings = []
        # for i in range(S.shape[0]):
        #     for j in range(i+1, S.shape[0]):
        #         for k in range(1,S.shape[1]):
        #             pairings.append([i,j,k])

        i, j, k = np.mgrid[0:S.shape[0], 0:S.shape[0], 1:S.shape[1]]
        valid_pairs_mask = i < j
        pairings = np.vstack((i[valid_pairs_mask], j[valid_pairs_mask], k[valid_pairs_mask])).T.tolist()

        random.shuffle(pairings)
        for pairing in pairings:
            doOneTrade(trade_volume,
                    S, net_exports, 
                    prices, 
                    pairing[2], 
                    pairing[1], 
                    pairing[0],
                    trade_chage=0.05)
        return trade_volume, net_exports

def doOneTrade(trade_volume, S, net_exports, prices, industry, c2, c1, trade_chage):
    difference = abs(prices[c1, industry] - prices[c2, industry])
    percentage_difference = (difference/prices[c1, industry]) * 100

    if difference < 0.01 or percentage_difference < 1:
        return

    if prices[c1, industry] < prices[c2, industry]:
        exporterName = c1; importerName = c2

    else:
        exporterName = c2; importerName = c1

    inventory_exporter = S[exporterName, industry]-net_exports[exporterName, industry]
    money_importer = S[importerName, 0] - net_exports[importerName, 0]

    target_trade_change = trade_chage
    money_exchanged = prices[exporterName, industry]

    if (inventory_exporter > target_trade_change) and (money_importer > money_exchanged):
        trade_volume[exporterName, importerName, industry] -= target_trade_change
        trade_volume[importerName, exporterName, industry] += target_trade_change

        trade_volume[exporterName, importerName, 0] += money_exchanged
        trade_volume[importerName, exporterName, 0] -= money_exchanged

        net_exports[exporterName, industry] += target_trade_change
        net_exports[importerName, industry] -= target_trade_change
        net_exports[exporterName, 0] -= money_exchanged
        net_exports[importerName, 0] += money_exchanged
