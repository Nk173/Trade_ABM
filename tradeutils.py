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



def resetAllMonetaryFlows(trade_volume, S, prices, tariffs):
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
                    trade_cost = -trade_volume[c1,c2,i]*prices[c1,i]*(1+tariffs[c1,c2,i])

                    if nominal_budget_left[c2] < abs(trade_cost):
                        trade_cost = nominal_budget_left[c2]
                        new_trade_volume = -trade_cost/prices[c1,i]*(1+tariffs[c1,c2,i])

                    # Update trade volumes and costs between the two countries.
                    trade_volume[c1,c2,i] = new_trade_volume
                    trade_volume[c2,c1,i] = -new_trade_volume
                    trade_volume[c1,c2,0] += trade_cost
                    trade_volume[c2,c1,0] -= trade_cost

                    # Update prices and remaining budget based on the recalibrated trade flows.
                    old_price = prices[c1,i]*(1+tariffs[c1,c2,i])
                    nominal_budget_left[c1] += (trade_cost + old_trade_volume * old_price)
                    nominal_budget_left[c2] -= (trade_cost + old_trade_volume * old_price)
    return trade_volume

#### PA

# def calculate_export_weights(S, net_exports):
#     """Calculate export weights for each country-product pair based on net exports."""
#     export_weights = np.maximum(net_exports, 0)  # Use only positive net exports
#     total_exports_per_product = np.sum(export_weights, axis=0)
    
#     # Avoid division by zero by replacing zeros with a small positive value
#     total_exports_per_product[total_exports_per_product == 0] = 1e-8
#     normalized_weights = export_weights / total_exports_per_product  # Normalize by total exports for each product
    
#     return normalized_weights

# def sample_pairs_with_preferential_attachment(S, net_exports):
#     """Generate trading pairs with preferential attachment based on export volume."""
#     n_countries, n_products = S.shape[0], S.shape[1]
    
#     # Calculate the export weights for preferential attachment
#     weights = calculate_export_weights(S, net_exports)
    
#     # Create all possible country pairs (c1, c2) for each product
#     i, j, k = np.mgrid[0:n_countries, 0:n_countries, 1:n_products]
#     valid_pairs_mask = i < j  # Ensure (c1, c2) is unique and avoids self-pairs
#     pairings = np.vstack((i[valid_pairs_mask], j[valid_pairs_mask], k[valid_pairs_mask])).T.tolist()

#     # Assign weights to each pair based on the exporting country's weight for the product
#     pair_weights = np.array([weights[pair[0], pair[2]] + weights[pair[1], pair[2]] for pair in pairings])

#     # Replace NaN or inf values with a small positive value and normalize
#     pair_weights = np.nan_to_num(pair_weights, nan=0.0, posinf=0.0, neginf=0.0)
#     total_weight = pair_weights.sum()
    
#     if total_weight == 0:
#         # If all weights are zero, assign equal probabilities to all pairs
#         pair_weights = np.ones_like(pair_weights)
#         total_weight = pair_weights.sum()
    
#     pair_weights /= total_weight  # Ensure the sum is exactly 1

#     # Filter pairs with non-zero weights
#     non_zero_indices = np.where(pair_weights > 0)[0]
#     non_zero_weights = pair_weights[non_zero_indices].astype(np.float64)  # Convert to float64
#     non_zero_pairings = [pairings[idx] for idx in non_zero_indices]

#     # Sample pairs based on the non-zero weights
#     sampled_indices = np.random.choice(len(non_zero_pairings), size=len(non_zero_pairings), replace=False, p=non_zero_weights)

#     # Return the sampled pairs in the order they were chosen
#     sampled_pairs = [non_zero_pairings[idx] for idx in sampled_indices]
    
#     return sampled_pairs


# def doAllTrades(trade_volume, S, prices, trade_change, tariffs):  
#     net_exports = compute_net_export(S, trade_volume)
    
#     # Use preferential attachment to get trading pairs
#     sampled_pairs = sample_pairs_with_preferential_attachment(S, net_exports)

#     # Process each trade pair
#     for c1, c2, industry in sampled_pairs:
#         doOneTrade(trade_volume, S, net_exports, prices, industry, c2, c1, trade_change, tariffs)
    
#     return trade_volume, net_exports
# ######


def doAllTrades(trade_volume, S, prices, trade_change, tariffs):  
        trade_volume = resetAllMonetaryFlows(trade_volume, S, prices, tariffs)
        net_exports = compute_net_export(S, trade_volume)

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
                    trade_change=trade_change,
                    tariffs=tariffs)
        return trade_volume, net_exports

def doOneTrade(trade_volume, S, net_exports, prices, industry, c2, c1, trade_change, tariffs):
    # Calculate effective prices including tariffs
    price_with_tariff_c1 = prices[c1, industry] * (1 + tariffs[c1, c2, industry])  # c2 imports from c1
    price_with_tariff_c2 = prices[c2, industry] * (1 + tariffs[c2, c1, industry])  # c1 imports from c2

    # Determine price differences and percentage differences
    difference = abs(price_with_tariff_c1 - price_with_tariff_c2)
    percentage_difference = (difference / min(price_with_tariff_c1, price_with_tariff_c2)) * 100

    # If the price difference is negligible, return
    if difference < 0.01 or percentage_difference < 1:
        return

    # Determine exporter and importer based on tariff-inclusive price advantage
    if price_with_tariff_c1 < price_with_tariff_c2:
        exporterName = c1
        importerName = c2
        effective_price = price_with_tariff_c1
    else:
        exporterName = c2
        importerName = c1
        effective_price = price_with_tariff_c2

    # Calculate available inventory and budget
    inventory_exporter = S[exporterName, industry] - net_exports[exporterName, industry]
    money_importer = S[importerName, 0] - net_exports[importerName, 0]

    # Calculate trade change and associated money exchanged
    target_trade_change = S[exporterName, industry] * trade_change
    base_money_exchanged = effective_price * target_trade_change

    # Ensure trade is feasible based on inventory and importer budget
    if (inventory_exporter > target_trade_change) and (money_importer > base_money_exchanged):
        
        # Adjust trade volumes
        trade_volume[exporterName, importerName, industry] -= target_trade_change
        trade_volume[importerName, exporterName, industry] += target_trade_change

        # Adjust monetary flows
        trade_volume[exporterName, importerName, 0] += base_money_exchanged
        trade_volume[importerName, exporterName, 0] -= base_money_exchanged

        # Update net exports
        net_exports[exporterName, industry] += target_trade_change
        net_exports[importerName, industry] -= target_trade_change
        net_exports[exporterName, 0] -= base_money_exchanged
        net_exports[importerName, 0] += base_money_exchanged
