import numpy as np
import random
import sys

def compute_net_export(S, trades):
    net_exports = np.zeros(S.shape, dtype=np.float128)
    mask = np.ones(S.shape[0], dtype=bool)
    for i in range(S.shape[0]):
        mask[i] = False
        net_exports[i] = -np.sum(trades[i, mask], axis=0)
        mask[i] = True

        # Ensure net exports stay within the bounds of [-inf, S]
        net_exports[i] = np.clip(net_exports[i], None, S[i])
    return net_exports

# def compute_net_export(S, trades):
#     total_imports = np.sum(trades, axis=0)
#     net_exports = -total_imports + trades
#     return net_exports



def resetAllMonetaryFlows(trade_volume, S, prices):
    net_exports = compute_net_export(S, trade_volume)
    export_adjustment = np.zeros((S.shape), dtype=np.float128)
    nominal_budget_left = np.zeros((S.shape[0]), dtype=np.float128)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if j > 0:
                np.seterr(divide='ignore', invalid='ignore')
                export_adjustment[i, j] = 1 if net_exports[i, j] >0 else min(1, S[i, j]/abs(net_exports[i,j]) + 0.001)
                if abs(export_adjustment[i,j])>1:
                    print('Error1')
                    sys.exit(0) 
                else:
                    nominal_budget_left[i] = S[i, 0] - net_exports[i,0]
                    # nominal_budget_left[i] = max(S[i, 0] - net_exports[i,0],0)

    for c1 in range(S.shape[0]):
        for c2 in range(S.shape[0]):
            trade_volume[c1,c2,0] = 0.0
            trade_volume = trade_volume.astype(np.float128)

    
    for i in range(1,S.shape[1]):
        for c1 in range(S.shape[0]):
            for c2 in range(S.shape[0]):
                if c1==c2:
                    continue
                old_trade_volume = trade_volume[c1,c2,i]

                if old_trade_volume < 0.0:
                    new_trade_volume = old_trade_volume * export_adjustment[c1,i]
                    trade_cost = -trade_volume[c1,c2,i]*prices[c1,i]

                    if nominal_budget_left[c2] < abs(trade_cost):
                        trade_cost = nominal_budget_left[c2]
                        new_trade_volume = -trade_cost/prices[c1,i]

                    # if abs(trade_cost)> 1e20 or abs(new_trade_volume)>1e20:
                    #     print(trade_cost, new_trade_volume, 'Error2')
                    #     continue

                    trade_volume[c1,c2,i] = new_trade_volume
                    trade_volume[c2,c1,i] = -new_trade_volume

                    trade_volume[c1,c2,0] += trade_cost
                    trade_volume[c2,c1,0] -= trade_cost

                    old_price = prices[c1,i]
                    nominal_budget_left[c1] += (trade_cost + old_trade_volume * old_price)
                    nominal_budget_left[c2] -= (trade_cost + old_trade_volume * old_price)


def doAllTrades(trade_volume, S, prices, trade_chage):  
        resetAllMonetaryFlows(trade_volume, S, prices)
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
