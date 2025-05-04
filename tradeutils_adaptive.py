import numpy as np
import random

class TradeNetwork:
    def __init__(self, num_countries, num_products):
        self.num_countries = num_countries
        self.num_products = num_products

        # Trade relationship matrix: stores past trade success scores
        self.trade_weights = np.ones((num_countries, num_countries, num_products), dtype=np.float64)

    def update_trade_weights(self, trade_volume, S, net_exports):
        """Update trade probabilities based on past trading success."""
        for c1 in range(self.num_countries):
            for c2 in range(self.num_countries):
                if c1 == c2:
                    continue  # Skip self-trade

                for industry in range(1, self.num_products):
                    trade_success = abs(trade_volume[c1, c2, industry])

                    # If a country had successful exports, increase its weight
                    if trade_success > 0:
                        self.trade_weights[c1, c2, industry] *= 1.1  # Reward success
                    else:
                        self.trade_weights[c1, c2, industry] *= 0.9  # Reduce unsuccessful trade

                    # Prevent runaway weights
                    self.trade_weights[c1, c2, industry] = min(max(self.trade_weights[c1, c2, industry], 0.1), 10)

    def get_adaptive_trade_pairs(self):
        """Generate trade pairs based on updated trade probabilities."""
        all_pairs = [
            (c1, c2, industry)
            for c1 in range(self.num_countries)
            for c2 in range(self.num_countries)
            for industry in range(1, self.num_products)
            if c1 != c2
        ]

        # Assign weights based on past trade success
        weights = np.array([
            self.trade_weights[c1, c2, industry] for (c1, c2, industry) in all_pairs
        ])

        # Normalize weights to sum to 1 for sampling
        weights /= np.sum(weights)

        # Sample trading pairs adaptively
        sampled_pairs = random.choices(all_pairs, weights=weights, k=len(all_pairs))

        return sampled_pairs

def compute_net_export(S, trades):
    """Calculate net exports for each country."""
    net_exports = np.zeros(S.shape, dtype=np.float128)
    mask = np.ones(S.shape[0], dtype=bool)

    for i in range(S.shape[0]):
        mask[i] = False
        net_exports[i] = -np.sum(trades[i, mask], axis=0)
        mask[i] = True
        net_exports[i] = np.clip(net_exports[i], None, S[i])

    return net_exports

def resetAllMonetaryFlows(trade_volume, S, prices):
    """Adjust trade flows dynamically based on past performance."""
    net_exports = compute_net_export(S, trade_volume)
    export_adjustment = np.zeros((S.shape), dtype=np.float128)
    nominal_budget_left = np.zeros((S.shape[0]), dtype=np.float128)

    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if j > 0:
                export_adjustment[i, j] = 1 if net_exports[i, j] > 0 else min(1, S[i, j] / abs(net_exports[i, j]) + 0.001)
                nominal_budget_left[i] = S[i, 0] - net_exports[i, 0]

    for c1 in range(S.shape[0]):
        for c2 in range(S.shape[0]):
            trade_volume[c1, c2, 0] = 0.0
            trade_volume = trade_volume.astype(np.float128)

    for i in range(1, S.shape[1]):
        for c1 in range(S.shape[0]):
            for c2 in range(S.shape[0]):
                if c1 == c2:
                    continue

                old_trade_volume = trade_volume[c1, c2, i]
                if old_trade_volume < 0.0:
                    new_trade_volume = old_trade_volume * export_adjustment[c1, i]
                    trade_cost = -trade_volume[c1, c2, i] * prices[c1, i]

                    if nominal_budget_left[c2] < abs(trade_cost):
                        trade_cost = nominal_budget_left[c2]
                        new_trade_volume = -trade_cost / prices[c1, i]

                    trade_volume[c1, c2, i] = new_trade_volume
                    trade_volume[c2, c1, i] = -new_trade_volume
                    trade_volume[c1, c2, 0] += trade_cost
                    trade_volume[c2, c1, 0] -= trade_cost

                    nominal_budget_left[c1] += (trade_cost + old_trade_volume * prices[c1, i])
                    nominal_budget_left[c2] -= (trade_cost + old_trade_volume * prices[c1, i])

    return trade_volume

def update_tariffs(tariffs, net_exports):
    """Adjust tariffs dynamically based on trade imbalances."""
    for c1 in range(tariffs.shape[0]):
        for c2 in range(tariffs.shape[1]):
            if c1 == c2:
                continue

            for industry in range(1, tariffs.shape[2]):
                trade_balance = net_exports[c1, industry]

                if trade_balance < 0:  # Trade deficit: increase tariff
                    tariffs[c1, c2, industry] *= 1.05
                else:  # Trade surplus: decrease tariff
                    tariffs[c1, c2, industry] *= 0.95

                tariffs[c1, c2, industry] = min(max(tariffs[c1, c2, industry], 0.01), 0.5)

def doOneTrade(trade_volume, S, net_exports, prices, industry, c2, c1, trade_change, tariffs):
    """Execute one trade transaction with dynamic pricing and tariffs."""
    price_with_tariff_c1 = prices[c1, industry] * (1 + tariffs[c1, c2, industry])
    price_with_tariff_c2 = prices[c2, industry] * (1 + tariffs[c2, c1, industry])

    difference = abs(price_with_tariff_c1 - price_with_tariff_c2)
    percentage_difference = (difference / min(price_with_tariff_c1, price_with_tariff_c2)) * 100

    if difference < 0.01 or percentage_difference < 1:
        return

    if price_with_tariff_c1 < price_with_tariff_c2:
        exporterName, importerName, effective_price = c1, c2, price_with_tariff_c1
    else:
        exporterName, importerName, effective_price = c2, c1, price_with_tariff_c2

    inventory_exporter = S[exporterName, industry] - net_exports[exporterName, industry]
    money_importer = S[importerName, 0] - net_exports[importerName, 0]

    target_trade_change = S[exporterName, industry] * trade_change
    base_money_exchanged = effective_price * target_trade_change

    if inventory_exporter > target_trade_change and money_importer > base_money_exchanged:
        trade_volume[exporterName, importerName, industry] -= target_trade_change
        trade_volume[importerName, exporterName, industry] += target_trade_change
        trade_volume[exporterName, importerName, 0] += base_money_exchanged
        trade_volume[importerName, exporterName, 0] -= base_money_exchanged

def doAllTrades(trade_network, trade_volume, S, prices, trade_change, tariffs):
    trade_volume = resetAllMonetaryFlows(trade_volume, S, prices)
    net_exports = compute_net_export(S, trade_volume)

    update_tariffs(tariffs, net_exports)
    trade_network.update_trade_weights(trade_volume, S, net_exports)
    sampled_pairs = trade_network.get_adaptive_trade_pairs()

    for c1, c2, industry in sampled_pairs:
        doOneTrade(trade_volume, S, net_exports, prices, industry, c2, c1, trade_change, tariffs)

    return trade_volume, net_exports
