import numpy as np
import pandas as pd
import pickle
import random
import sys, time
from functions import production_function, regularise, wage_function, demand_function, innovate, generate_nested_matrix, visualize_multi_layer_trade_network, array_to_dataframe
from tradeutils_adaptive import doAllTrades
from pricing import updatePricesAndConsume
from tqdm import tqdm

# np.random.seed(1)

def convert_to_matrices(params, n_countries, n_products, default_value=0):
    alpha_matrix = np.zeros((n_countries, n_products))
    beta_matrix = np.zeros((n_countries, n_products))
    A_matrix = np.zeros((n_countries, n_products))

    for i in range(n_countries):
        for j in range(n_products):
            alpha_key = f'alpha-{i}-{j}'
            A_key = f'A-{i}-{j}'
            beta_key = f'beta-{i}-{j}'
            alpha_matrix[i, j] = params.get(alpha_key, default_value)
            beta_matrix[i, j] = params.get(beta_key, default_value)
            A_matrix[i, j] = params.get(A_key, default_value)

    return alpha_matrix, beta_matrix, A_matrix

# Example usage with the same params and assuming the missing keys
# alpha, beta, A = convert_to_matrices(params, n_countries, n_products)

def gulden_vectorised(case, seed, n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta, share, tariffs, psi, iterations=2000, Tr_time=1, tariff_time=1, trade_change=1, autarky_time=10000,pricing_algorithm='dgp', utility_algorithm='geometric', prod_function='C-D', wage_algorithm='marginal_product', csv=False, plot=False, plot_simple=False, plot_anim=False,shock=None, shock_time=10000,  cm_time=10000, d=0.000, p_reassess=None, innovation=False, innovation_time=10000, gamma=1, eta=0.01,weights=None, elasticities=None, sigma=None, cnames=None, pnames=None):
    
    np.random.seed(seed=seed)

    ## Citizen-level
    # Initialize prices for each good in each nation
    prices = np.ones((n_countries, n_products))
    prices_vec = np.ones((n_countries, n_products, iterations))

    # Initialize industry choices for labor and capital for each citizen in each nation
    labor_choices = [np.zeros(citizens, dtype=int) for citizens in citizens_per_nation]
    capital_choices = [np.zeros((citizens, 2), dtype=int) for citizens in citizens_per_nation]
    income = [np.zeros(citizens) for citizens in citizens_per_nation]

    domestic_income = np.zeros((n_countries, n_countries))
    foreign_income = np.zeros((n_countries, n_countries))

    # Allow capital choices to take both industry and nation if capital mobility is enabled
    for nation in range(n_countries):
        labor_choices[nation] = np.random.choice(range(n_products), citizens_per_nation[nation])
        capital_choices[nation][:, 0] = np.random.choice(range(n_products), citizens_per_nation[nation])
        capital_choices[nation][:, 1] = np.ones(citizens_per_nation[nation]) * nation

    ## Nation-level
    # Calculate wages and returns to capital for each industry in each nation
    L = np.zeros((n_countries, n_products))
    K = np.zeros((n_countries, n_products))
    Q = np.zeros((n_countries, n_products))
    W = np.zeros((n_countries, n_products))
    R = np.zeros((n_countries, n_products))
    D = np.zeros((n_countries, n_products))
    D1 = np.zeros((n_countries, n_products))
    S = np.zeros((n_countries, n_products))
    net_exports = np.zeros(S.shape, dtype=np.float128)

    tv = np.zeros((n_countries, n_countries, n_products))
    labor = np.zeros((n_countries, n_products, iterations))
    capital = np.zeros((n_countries, n_products, iterations))
    production = np.zeros((n_countries, n_products, iterations))
    demand = np.zeros((n_countries, n_products, iterations))
    tariffs_rates = tariffs.copy()
    tariffs = np.zeros((n_countries, n_countries, n_products))

    supply = np.zeros((n_countries, n_products, iterations))
    net_exports_history = np.zeros((n_countries, n_products, iterations))
    trades_history = np.zeros((n_countries*n_products, n_countries, iterations))
    wage = np.zeros((n_countries, n_products, iterations))
    returns = np.zeros((n_countries, n_products, iterations))
    utility = np.zeros((n_countries, iterations))
    gdp_vec = np.zeros((n_countries, iterations))
    gnp_vec = np.zeros((n_countries, iterations))
    io_mat = np.zeros((n_countries, n_countries, n_products))
    if p_reassess == None:
        p_reassess=np.ones(n_countries)*0.004
    
    try:
        if len(Tr_time) > 1:
            T0 = Tr_time[0]
            T1 = Tr_time[1]
    except:
        T0 = Tr_time
        T1 = 0
    
    try:
        if len(shock_time)>1:
            S0 = shock_time[0]
            S1 = shock_time[1]
            shock_1 = shock[0]
            shock_2 = shock[1]

    
    except:
        S0=shock_time
        S1 = 0
        shock_1 = shock

    for t in tqdm(range(iterations)):

        Tr = False
        cap_mobility = False

        if t >= T0 or (T1 > 0 and t >= T1):
            Tr = True
            
        if t>=tariff_time:
            tariffs=tariffs_rates
        # else:
            # tariffs=tariffs

        if t >= autarky_time:
            if (T1 > 0 and t < T1):
                Tr = False

        if t >=S0:
            if S1>0 and t<S1:
                A = shock_1
                
            elif S1>0 and t> S1:
                A = shock_2
                
            elif S1==0:
                A = shock
            
        if t >= cm_time:
            cap_mobility = True

        # Identify the top 10% of workers globally
        all_worker_wages = np.concatenate( [W[n, labor_choices[n]] +R[n, labor_choices[n]] for n in range(n_countries)])  # Collect all wages
        if len(all_worker_wages) > 0:
            global_wage_threshold = np.percentile(all_worker_wages, 50)  # Find the 90th percentile wage
        else:
            global_wage_threshold = 0  # Default if no workers exist
        
        # Define ψ at a country-sector level (n_countries x n_products)
        psi_country_sector = np.zeros((n_countries, n_products))  # Start at 1 for all country-sector pairs
    
        for nation in range(n_countries):
            # Loop over each nation in the simulation.

            # Initialize arrays to determine which citizens will change their labor or capital choices.
            should_change_labor = np.zeros(citizens_per_nation[nation])
            should_change_capital = np.zeros(citizens_per_nation[nation])

            # Citizen update
            # Re-assess labor and capital choices

            # Find the industry with the maximum wage in the current nation.
            max_wage_industry = np.argmax(W[nation, :])
            # For each citizen, generate a random event (1 or 0) with a probability of 0.004.
            event_occurs = np.random.binomial(1, p_reassess[nation], size=citizens_per_nation[nation])

            # Determine which citizens should change labor based on the random event and wage comparison.
            # wages need to be d% above and the event needs to be true
            should_change_labor = (event_occurs == 1) * (
                        W[nation, labor_choices[nation]] < W[nation, max_wage_industry] * (1 + d))

            # Identify workers earning above the global threshold
            high_wage_workers = (W[nation, labor_choices[nation]] + R[nation, labor_choices[nation]])  >= global_wage_threshold
            
            
            # If a high-wage worker reassesses, increase ψ (efficiency)
            for worker in np.where(should_change_labor & high_wage_workers)[0]:  
                sector = labor_choices[nation][worker]  # Get the sector the worker is moving into
                psi_country_sector[nation, sector] = min(psi_country_sector[nation, sector]+ psi, 100)  # ✅ Cap at 2

            # Update the labor choices of citizens who decided to change.
            labor_choices[nation][should_change_labor == 1] = max_wage_industry

            # If capital mobility is enabled (cap_mobility is True).
            if cap_mobility:
                # Determine the industry with the maximum return globally (across all nations).
                max_return_industry = np.unravel_index(np.argmax(R, axis=None), R.shape)
                # Determine which citizens change capital choice based on random event and returns comparison.
                should_change_capital = (event_occurs == 1) * (
                            R[capital_choices[nation][:, 1], capital_choices[nation][:, 0]] < R[max_return_industry] * (1 + d))
                # Update the capital choices of citizens who decided to change.
                capital_choices[nation][should_change_capital == 1] = (max_return_industry[1], max_return_industry[0])

            else:
                # If capital mobility is disabled, only consider the current nation for max returns.
                max_return_industry = np.argmax(R[nation, :])
                # Determine which citizens should change their capital choice based on the random event and returns comparison.
                should_change_capital = (event_occurs == 1) * (
                            R[nation, capital_choices[nation][:, 0]] < R[nation, max_return_industry] * (1 + d))
                # Update the capital choices of citizens who decided to change.
                capital_choices[nation][should_change_capital == 1] = (max_return_industry, nation)

            # Calculate the total income for the nation by summing wages and returns for each citizen
            income[nation] = sum(
                W[nation, labor_choices[nation]] + R[capital_choices[nation][:, 1], capital_choices[nation][:, 0]])

            # Create a mask to identify foreign investments (other nations) for each citizen
            foreign_investment_nations = capital_choices[nation][:, 1] != nation
            domestic_investment_nations = capital_choices[nation][:,1] == nation

            # Calculate foreign income based on foreign investments (in the current nation)
            foreign_investment_indices = capital_choices[nation][:, 0][foreign_investment_nations]
            foreign_investment_nations = capital_choices[nation][:, 1][foreign_investment_nations]

            domestic_investment_indices = capital_choices[nation][:, 0][domestic_investment_nations]
            domestic_investment_nations = capital_choices[nation][:, 1][domestic_investment_nations]

            foreign_income[nation, foreign_investment_nations] = np.sum(R[foreign_investment_nations, foreign_investment_indices])
            domestic_income[nation, domestic_investment_nations] = np.sum(W[nation, labor_choices[nation]]) + np.sum(R[domestic_investment_nations, domestic_investment_indices])

        for nation in range(n_countries):
            for industry in range(n_products):
                # Loop over each industry to update the labor, capital, and demand for the nation.

                # Initialize labor and demand for the industry.
                L[nation, industry] = 0

                # D[nation, industry] = income[nation] / (n_products * prices[nation, industry])
                # Compute effective price for domestic and foreign demand
                effective_price_domestic = prices[nation, industry]
                effective_price_foreign = np.array([
                    prices[otherNation, industry] * (1 + tariffs[otherNation, nation, industry])
                    for otherNation in range(n_countries)
                ])

                # Domestic demand
                domestic_demand = demand_function(domestic_income[nation, nation], n_products, effective_price_domestic, method='hyperbolic', sigma=sigma, P0=100)

                # Foreign demand aggregated over all other nations
                foreign_demand = np.sum([
                    demand_function(foreign_income[otherNation, nation], n_products, effective_price_foreign[otherNation], method='hyperbolic', sigma=sigma, P0=100)
                    for otherNation in range(n_countries)
                ])


                # Demand for nation's goods = domestic demand + foreign demand
                D1[nation, industry] = domestic_demand + foreign_demand
#                 D1[nation, industry] = (domestic_income[nation,nation]/ (n_products*prices[nation,industry])) + np.sum([foreign_income[otherNation, nation]/ (n_products* prices[nation, industry]) for otherNation in range(n_countries)])

                # Update demand for the industry.
               

                # Update the labor count in the industry.
                L[nation, industry] = sum(labor_choices[nation] == industry)
#                 effective_L = L[nation, industry]
                effective_L = np.exp(psi_country_sector[nation, industry]) * sum(labor_choices[nation] == industry)


                ## should probably be moved OUT of the loop, or it will not wait for other
                ## citizens to make their choices
                K[nation, industry] = 0
                ## let's use a single call for now;  less code duplication matters more than small speed-up for debugging
                K[nation, industry] = sum(
                    ((capital_choices[c][i, 0] == industry) and (capital_choices[c][i, 1] == nation)) for c in
                    range(n_countries) for i in range(citizens_per_nation[c]))
                
                if K[nation, industry] < 1:
                    K[nation, industry] = 1

                # Compute the production, wages, and returns for the industry using some functions.

                Q[nation, industry] = production_function(A[nation, industry], effective_L, K[nation, industry],
                                                          alpha[nation, industry], beta[nation, industry], func=prod_function)
                
                W[nation, industry], R[nation, industry] = wage_function(A[nation, industry], effective_L,
                                                                         K[nation, industry], alpha[nation, industry],
                                                                         beta[nation, industry],
                                                                         p=prices[nation, industry],
                                                                         algorithm=wage_algorithm,share=share[nation])
                S[nation, industry] = Q[nation, industry]

        # Trade update
        if Tr:
            tv, net_exports = doAllTrades(tv, S, prices, trade_change, tariffs) 
            S = Q - net_exports
            io = array_to_dataframe(tv, (countries, countries, products))
            io = -io
            io[io<0] = 0
            for i in io.index.get_level_values(0).unique():
                io.loc[(i,slice(None)),i] = S[int(i),:]
            trades_history[:, :, t] = io.values

#         net_exports_history[:, :, t] = net_exports
        

        # Update prices and Consume
        prices, UT = updatePricesAndConsume(prices, D1, S, tariffs, pricing_algorithm, utility_algorithm, weights=weights, elasticities=elasticities, sigma=sigma)
        prices_vec[:, :, t] = prices
        utility[:, t] = UT
        gdp_vec[:, t] = income
        supply[:, :, t] = S
        demand[:,:, t] = D1
        labor[:,:,t] = L
        capital[:,:,t] = K
        production[:,:,t] = Q
        wage[:,:,t] = W
        returns[:,:,t] = R

    # variables to return:
    import pandas as pd
#     io_mat = np.sum(trades_history, axis=2)
    stabletime=500
    io_mat = np.sum(trades_history[:,:,stabletime:], axis=2)
    io_df = pd.DataFrame(io_mat, index=io.index, columns=io.columns)
    
    io_df2 = io_df[~io_df.index.get_level_values(1).isin([0])]
    
    io_df2 = io_df2.reset_index()
    io_df2 = io_df2.rename(columns={io_df2.columns[0]:'Group',io_df2.columns[1]:'Sector'})
    io_df2 = io_df2.set_index(['Group', 'Sector'])
    
    io_df = io_df.reset_index()
    io_df = io_df.rename(columns={io_df.columns[0]:'Group',io_df.columns[1]:'Sector'})
    io_df = io_df.set_index(['Group', 'Sector'])
    
    io_dfval = io_df
    io_df2val = io_df2
    
    io_df = io_df.div(io_df.sum(axis=1), axis=0)
    io_df2 = io_df2.div(io_df2.sum(axis=1), axis=0)
#     total_exports=np.sum(net_exports_history, axis=2)
    # io = array_to_dataframe(trades, (countries, countries, products))
    # io = -io
    # io[io<0]=0
    # s = np.sum(supply, axis=2)
    # for i in io.index.get_level_values(0).unique():
    #     io.loc[(i,slice(None)),i] = s[int(i),:]

    # Save the results of Labor, Productions, Wages, and Returns to a csv file
    if csv:
        import pandas as pd
        for nation in range(n_countries):
            for industry in range(n_products):
                df = pd.DataFrame({'t': range(iterations),
                                   'labor': labor[nation, industry, :],
                                   'capital': capital[nation, industry, :],
                                   'production': production[nation, industry, :],
                                   'wages': wage[nation, industry, :],
                                   'returns': returns[nation, industry, :],
                                   'prices': prices_vec[nation, industry, :],
                                   'demand': demand[nation, industry, :],
                                   'utility': utility[nation, :]})
                df.to_csv('vectorised_{}_{}.csv'.format(countries[nation], products[industry]))
                
        # Save the 4-D array as a pickle file
        with open('trade.pickle', 'wb') as pickle_file:
            pickle.dump(trades_history, pickle_file)

        # Plot the results
    if plot_simple:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['axes.titlesize'] = 20

        color_map = cm.get_cmap('tab20', n_countries * n_products)
        color_list = [color_map(i) for i in range(n_countries * n_products)]
        line_styles = ['-', '--', '-.', ':']
        color_dict = {(n, p): color_list[i] for i, (n, p) in enumerate([(n, p) for n in range(n_countries) for p in range(n_products)])}
        highlight_combinations = [(4, 1), (0, 1), (3, 1)]
        highlight_countries = [0, 3, 4]

        fig0, ax0 = plt.subplots(2, 4, figsize=(20, 10))
        fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))

        handles_production, labels_production = [], []
        handles_gdp, labels_gdp = [], []

        for nation in range(n_countries):
            for industry in range(n_products):
                label = f'{cnames[nation]}-{pnames[industry]}'
                color = color_dict[(nation, industry)]
                linestyle = line_styles[nation % len(line_styles)]
                alpha_value = 1.0 if (nation, industry) in highlight_combinations else 0.3

                for i, (data, title) in enumerate(zip(
                    [production, demand, labor, wage, prices_vec, supply, capital, returns],
                    ['Production', 'Demand', 'Labor', 'Wages', 'Prices', 'Supply', 'Capital', 'Returns']
                )):
                    line, = ax0[i//4, i%4].plot(range(iterations), data[nation, industry, :],
                                                label=label if (nation, industry) in highlight_combinations else None,
                                                color=color, linestyle=linestyle, alpha=alpha_value)
                    ax0[i//4, i%4].axvline(x=T0, linestyle=':', color='k', label='Trade Opens' if i == 0 else "")
                    ax0[i//4, i%4].axvline(x=tariff_time, linestyle=':', color='r', label='Tariff Applied' if i == 0 else "")

                    ax0[i//4, i%4].set_title(title)
                if (nation, industry) in highlight_combinations:
                        handles_production.append(line)
                        labels_production.append(label)
                    

            line_gdp, = ax1[0].plot(range(iterations), (gdp_vec[nation, :] / citizens_per_nation[nation]), 
                                label=cnames[nation] if nation in highlight_countries else None,
                                color=color_list[nation], linestyle='-', alpha=1.0 if nation in highlight_countries else 0.3)
        
            if nation in highlight_countries:
                handles_gdp.append(line_gdp)
                labels_gdp.append(cnames[nation])

            ax1[1].plot(range(iterations), utility[nation, :], label=cnames[nation] if nation in highlight_countries else None,
                        color=color_list[nation], linestyle='-', alpha=1.0 if nation in highlight_countries else 0.3)
            ax1[0].axvline(x=T0, linestyle=':', color='k', label='Trade Opens' if i == 0 else "")
            ax1[0].axvline(x=tariff_time, linestyle=':', color='r', label='Tariff Applied' if i == 0 else "")
            ax1[1].axvline(x=T0, linestyle=':', color='k', label='Trade Opens' if i == 0 else "")
            ax1[1].axvline(x=tariff_time, linestyle=':', color='r', label='Tariff Applied' if i == 0 else "")

    
        ax1[0].set_title('GDP per Capita')
        ax1[0].set_ylabel('GDP per Capita')
        ax1[1].set_title('Utility')
        ax1[1].set_ylabel('Utility')
        
        for i in range(2):
            for j in range(4):
                ax0[i, j].legend(handles_production, labels_production, loc='upper right', fontsize=12, frameon=True)

        # Label counter for subplots
        subplot_labels = iter(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

        for i in range(2):
            for j in range(4):
                label = next(subplot_labels)
                ax0[i, j].text(0.95, 0.75, f'({label})', transform=ax0[i, j].transAxes,
                              fontsize=12, va='top', ha='left', weight='bold')
        
        # Add labels to ax2 plots
        ax1[0].text(0.95, 0.75, '(i)', transform=ax1[0].transAxes, fontsize=12, va='top', ha='left', weight='bold')
        ax1[1].text(0.95, 0.75, '(j)', transform=ax1[1].transAxes, fontsize=12, va='top', ha='left', weight='bold')

        ax1[0].legend(handles_gdp, labels_gdp, loc='upper right', fontsize=10, frameon=True)
        ax1[1].legend(handles_gdp, labels_gdp, loc='upper right', fontsize=10, frameon=True)
        
        fig0.tight_layout()
        fig1.tight_layout()
        
        fig0.savefig(f'{case}_simple_cleaned.png')
        fig1.savefig(f'{case}_gdp_simple.png')

    # Plot the results
    if plot_anim:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16

        variables = {
            'Production': production,
            'Demand': demand,
            'Labor': labor,
            'Wages': wage,
            'Prices': prices_vec,
            'Supply': supply,
            'Capital': capital,
            'Returns': returns
        }

        lines = {name: [] for name in variables}
        x = list(range(iterations))
        highlight_combinations = [(4, 1), (0, 1), (3, 1)]
        highlight_countries = [0, 3, 4]

        for i, (varname, var) in enumerate(variables.items()):
            row, col = divmod(i, 4)
            ax[row, col].set_title(varname)
            ax[row, col].set_xlim(0, iterations)
            ax[row, col].set_ylim(0, np.max(var) * 1.1)

            for n in range(n_countries):
                for p in range(n_products):
                    label = f'{cnames[n]}-{pnames[p]}'
                    alpha = 1.0 if (n, p) in highlight_combinations else 0.3
                    (line,) = ax[row, col].plot([], [], label=label if alpha == 1.0 else None,
                                                color=f"C{n}", alpha=alpha)
                    lines[varname].append((line, n, p))

        lines_gdp = []
        lines_util = []

        for n in range(n_countries):
            alpha = 1.0 if n in highlight_countries else 0.3
            (gdp_line,) = ax2[0].plot([], [], label=cnames[n], color=f"C{n}", alpha=alpha)
            (util_line,) = ax2[1].plot([], [], label=cnames[n], color=f"C{n}", alpha=alpha)
            lines_gdp.append((gdp_line, n))
            lines_util.append((util_line, n))

        ax2[0].set_title("GDP per Capita")
        ax2[0].set_xlim(0, iterations)
        ax2[0].set_ylim(0, np.max(gdp_vec / np.array(citizens_per_nation)[:, None]) * 1.1)
        ax2[1].set_title("Utility")
        ax2[1].set_xlim(0, iterations)
        ax2[1].set_ylim(0, np.max(utility) * 1.1)

        def init():
            for varlines in lines.values():
                for line, _, _ in varlines:
                    line.set_data([], [])
            for line, _ in lines_gdp + lines_util:
                line.set_data([], [])
            return sum([list(map(lambda x: x[0], l)) for l in lines.values()], []) + \
                [l[0] for l in lines_gdp + lines_util]

        def update(frame):
            for varname, varlines in lines.items():
                var = variables[varname]
                for line, n, p in varlines:
                    line.set_data(x[:frame], var[n, p, :frame])

            for line, n in lines_gdp:
                line.set_data(x[:frame], gdp_vec[n, :frame] / citizens_per_nation[n])
            for line, n in lines_util:
                line.set_data(x[:frame], utility[n, :frame])

            return sum([list(map(lambda x: x[0], l)) for l in lines.values()], []) + \
                [l[0] for l in lines_gdp + lines_util]

        ani = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=iterations, interval=50, blit=False)

        ani.save(f"{case}_full_animation.mp4", writer='ffmpeg', dpi=150)
        print(f"Saved animation to {case}_full_animation.mp4")

    if plot:
        import matplotlib.pyplot as plt
        variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI', 'MRS']
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['axes.titlesize'] = 20
        
        # Label counter for subplots
        subplot_labels = iter(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

        for i in range(2):
            for j in range(4):
                label = next(subplot_labels)
                ax[i, j].text(0.95, 0.85, f'({label})', transform=ax[i, j].transAxes,
                              fontsize=18, va='top', ha='right', weight='bold')
                
        for nation in range(n_countries):
            for industry in range(n_products):
                ax[0, 0].plot(range(iterations), production[nation, industry, :],
                              label='{}-{}'.format(cnames[nation] if cnames else countries[nation], 
                                                   pnames[industry] if pnames else products[industry]))
                ax[0, 0].set_title('Production')

                ax[1, 0].plot(range(iterations), prices_vec[nation, industry, :],
                              label='{}-{}'.format(cnames[nation] if cnames else countries[nation], 
                                                   pnames[industry] if pnames else products[industry]))
                ax[1, 0].set_title('Prices')

                ax[0, 1].plot(range(iterations), demand[nation, industry, :],
                              label='{}-{}'.format(cnames[nation] if cnames else countries[nation], 
                                                   pnames[industry] if pnames else products[industry]))
                ax[0, 1].set_title('Demand')

                ax[1, 1].plot(range(iterations), supply[nation, industry, :],
                              label='{}-{}'.format(cnames[nation] if cnames else countries[nation], 
                                                   pnames[industry] if pnames else products[industry]))
                ax[1, 1].set_title('Supply')

                ax[0, 2].plot(range(iterations), labor[nation, industry, :],
                              label='{}-{}'.format(cnames[nation] if cnames else countries[nation], 
                                                   pnames[industry] if pnames else products[industry]))
                ax[0, 2].set_title('Labor')

                ax[1, 2].plot(range(iterations), capital[nation, industry, :],
                              label='{}-{}'.format(cnames[nation] if cnames else countries[nation], 
                                                   pnames[industry] if pnames else products[industry]))
                ax[1, 2].set_title('Capital')

                ax[0, 3].plot(range(iterations), wage[nation, industry, :],
                              label='{}-{}'.format(cnames[nation] if cnames else countries[nation], 
                                                   pnames[industry] if pnames else products[industry]))
                ax[0, 3].set_title('Wages')

                ax[1, 3].plot(range(iterations), returns[nation, industry, :],
                              label='{}-{}'.format(cnames[nation] if cnames else countries[nation], 
                                                   pnames[industry] if pnames else products[industry]))
                ax[1, 3].set_title('Returns')

            ax2[0].plot(range(iterations), (gdp_vec[nation, :] / citizens_per_nation[nation]),
                          label='{}'.format(cnames[nation] if cnames else countries[nation]))
            ax2[0].set_title('GDP per capita', )


            ax2[1].plot(range(iterations), utility[nation, :], label='{}'.format(cnames[nation] if cnames else countries[nation]))
#             ax2[1].legend(loc='upper left', fontsize=10)  # Legend for Utility
            ax2[1].set_title('Utility')
            ax2[1].set_xlabel('Time')
            ax2[1].set_ylabel('Utility')
            

            if T1 > 0:
                T0 = Tr_time[0]
                T1 = Tr_time[1]
                AT = autarky_time
                ax[0,0].axvline(x=T0, ls= '--', color='k')
                ax[1,0].axvline(x=T0, ls= '--', color='k')
                ax[0,1].axvline(x=T0, ls= '--', color='k')
                ax[1,1].axvline(x=T0, ls= '--', color='k')
                ax[0,2].axvline(x=T0, ls= '--', color='k')
                ax[1,2].axvline(x=T0, ls= '--', color='k')
                ax[0,3].axvline(x=T0, ls= '--', color='k')
                ax[1,3].axvline(x=T0, ls= '--', color='k')
                ax2[0].axvline(x=T0, ls= '--', color='k')
                ax2[1].axvline(x=T0, ls= '--', color='k')
            
                ax[0,0].axvline(x=T1, ls= '--', color='k')
                ax[1,0].axvline(x=T1, ls= '--', color='k')
                ax[0,1].axvline(x=T1, ls= '--', color='k')
                ax[1,1].axvline(x=T1, ls= '--', color='k')
                ax[0,2].axvline(x=T1, ls= '--', color='k')
                ax[1,2].axvline(x=T1, ls= '--', color='k')
                ax[0,3].axvline(x=T1, ls= '--', color='k')
                ax[1,3].axvline(x=T1, ls= '--', color='k')
                ax2[0].axvline(x=T1, ls= '--', color='k')
                ax2[1].axvline(x=T1, ls= '--', color='k')
                
                ax[0,0].axvline(x=AT, ls= '--', color='k')
                ax[1,0].axvline(x=AT, ls= '--', color='k')
                ax[0,1].axvline(x=AT, ls= '--', color='k')
                ax[1,1].axvline(x=AT, ls= '--', color='k')
                ax[0,2].axvline(x=AT, ls= '--', color='k')
                ax[1,2].axvline(x=AT, ls= '--', color='k')
                ax[0,3].axvline(x=AT, ls= '--', color='k')
                ax[1,3].axvline(x=AT, ls= '--', color='k')
                ax2[0].axvline(x=AT, ls= '--', color='k')
                ax2[1].axvline(x=AT, ls= '--', color='k')
#                 ax2[0].annotate('Trade', xy=(T0, ax2[0].get_ylim()[0]), 
#                     xytext=(T0, ax2[0].get_ylim()[0] - (ax2[0].get_ylim()[1] - ax2[0].get_ylim()[0]) * 0.05), rotation=90, va='center', ha='center', fontsize=10, color='black')
                
#                 ax2[0].annotate('Trade', xy=(T1, ax2[0].get_ylim()[0]), 
#                     xytext=(T1, ax2[0].get_ylim()[0] - (ax2[0].get_ylim()[1] - ax2[0].get_ylim()[0]) * 0.05), rotation=90, va='center', ha='center', fontsize=10, color='black')
                
#                 ax2[1].annotate('Trade', xy=(T0, ax2[1].get_ylim()[0]), 
#                     xytext=(T0, ax2[1].get_ylim()[0] - (ax2[1].get_ylim()[1] - ax2[1].get_ylim()[0]) * 0.05), rotation=90, va='center', ha='center', fontsize=10, color='black')
                
#                 ax2[1].annotate('Trade', xy=(T1, ax2[1].get_ylim()[0]), 
#                     xytext=(T1, ax2[1].get_ylim()[0] - (ax2[1].get_ylim()[1] - ax2[1].get_ylim()[0]) * 0.05), rotation=90, va='center', ha='center', fontsize=10, color='black')
                
            else:
                ax[0,0].axvline(x=T0, ls= '--', color='k')
                ax[1,0].axvline(x=T0, ls= '--', color='k')
                ax[0,1].axvline(x=T0, ls= '--', color='k')
                ax[1,1].axvline(x=T0, ls= '--', color='k')
                ax[0,2].axvline(x=T0, ls= '--', color='k')
                ax[1,2].axvline(x=T0, ls= '--', color='k')
                ax[0,3].axvline(x=T0, ls= '--', color='k')
                ax[1,3].axvline(x=T0, ls= '--', color='k')
                ax2[0].axvline(x=T0, ls= '--', color='k')
                ax2[1].axvline(x=T0, ls= '--', color='k')
#                 ax2[0].annotate('Trade', xy=(T0, ax2[0].get_ylim()[0]+0.05), 
#                     xytext=(T0+0.02, ax2[0].get_ylim()[0] + (ax2[0].get_ylim()[1] - ax2[0].get_ylim()[0]) * 0.5), rotation=90, va='center', ha='center', fontsize=10, color='black')
                                
#                 ax2[1].annotate('Trade', xy=(T0, ax2[1].get_ylim()[0]+0.05), 
#                     xytext=(T0+0.02, ax2[1].get_ylim()[0] + (ax2[1].get_ylim()[1] - ax2[1].get_ylim()[0]) * 0.5), rotation=90, va='center', ha='center', fontsize=10, color='black')                
                
            if t>shock_time:
                ax[0, 0].axvline(x=shock_time, ls= '--', color='k')
                ax[1, 0].axvline(x=shock_time, ls= '--', color='k')
                ax[0, 1].axvline(x=shock_time, ls= '--', color='k')
                ax[1, 1].axvline(x=shock_time, ls= '--', color='k')
                ax[0, 2].axvline(x=shock_time, ls= '--', color='k')
                ax[1, 2].axvline(x=shock_time, ls= '--', color='k')
                ax[0, 3].axvline(x=shock_time, ls= '--', color='k')
                ax[1, 3].axvline(x=shock_time, ls= '--', color='k')
                ax2[0].axvline(x=shock_time, ls= '--', color='k')
                ax2[1].axvline(x=shock_time, ls= '--', color='k')
#                 ax2[0].annotate('Shock', xy=(shock_time, ax2[0].get_ylim()[0]+0.05), 
#                     xytext=(shock_time+0.02, ax2[0].get_ylim()[0] + (ax2[0].get_ylim()[1] - ax2[0].get_ylim()[0]) * 0.5), rotation=90, va='center', ha='center', fontsize=10, color='black')
                                
#                 ax2[1].annotate('Shock', xy=(shock_time, ax2[1].get_ylim()[0]+0.05), 
#                     xytext=(shock_time+0.02, ax2[1].get_ylim()[0] + (ax2[1].get_ylim()[1] - ax2[1].get_ylim()[0]) * 0.5), rotation=90, va='center', ha='center', fontsize=10, color='black')
            
            if t>cm_time:
                ax[0, 0].axvline(x=cm_time, ls= '--', color='k')
                ax[1, 0].axvline(x=cm_time, ls= '--', color='k')
                ax[0, 1].axvline(x=cm_time, ls= '--', color='k')
                ax[1, 1].axvline(x=cm_time, ls= '--', color='k')
                ax[0, 2].axvline(x=cm_time, ls= '--', color='k')
                ax[1, 2].axvline(x=cm_time, ls= '--', color='k')
                ax[0, 3].axvline(x=cm_time, ls= '--', color='k')
                ax[1, 3].axvline(x=cm_time, ls= '--', color='k')
                ax2[0].axvline(x=cm_time, ls= '--', color='k')
                ax2[1].axvline(x=cm_time, ls= '--', color='k')
#                 ax2[0].annotate('cap mob', xy=(cm_time, ax2[0].get_ylim()[0]), 
#                     xytext=(cm_time, ax2[0].get_ylim()[0] - (ax2[0].get_ylim()[1] - ax2[0].get_ylim()[0]) * 0.05), rotation=90, va='center', ha='center', fontsize=10, color='black')
                                
#                 ax2[1].annotate('cap mob', xy=(cm_time, ax2[1].get_ylim()[0]), 
#                     xytext=(cm_time, ax2[1].get_ylim()[0] - (ax2[1].get_ylim()[1] - ax2[1].get_ylim()[0]) * 0.05), rotation=90, va='center', ha='center', fontsize=10, color='black')
            
        # Add labels to ax2 plots
        ax2[0].text(0.95, 0.85, '(i)', transform=ax2[0].transAxes, fontsize=18, va='top', ha='right', weight='bold')
        ax2[1].text(0.95, 0.85, '(j)', transform=ax2[1].transAxes, fontsize=18, va='top', ha='right', weight='bold')

        # Adjust layout to place fig2 below fig1
        fig.tight_layout()
        fig2.tight_layout()
        fig.subplots_adjust(bottom=0.15)  # Adjust this to leave enough space for the legend
        fig2.subplots_adjust(top=0.95)

        # Combine legends and place them centrally
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=n_countries, fontsize=15)

        handles2, labels2 = ax2[0].get_legend_handles_labels()
        fig2.legend(handles2, labels2, loc='lower center', bbox_to_anchor=(0.5, -0.015), ncol=n_countries, fontsize=13)
        
#         ax2[0].set_yscale('log')
        # Save the figures
        fig.savefig(f'{case}.png')
        fig2.savefig(f'{case}_gdp.png')

    return production, income, net_exports, io_df, io_df2, io_dfval, io_df2val

