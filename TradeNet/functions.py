# Functions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd

def production_function(A, L, K, alpha, beta,func='C-D'):
    if func=='C-D':
        Q = A * ((L)**(alpha)) * ((K)**(beta))
    elif func=='Leontif':
        L_prod = np.dot(np.dot(alpha,A), L)
        K_prod = np.dot(np.dot(alpha,A), L)
        Q = np.minimum(L_prod, K_prod)
    return  Q

# alpha = np.array([[0.5,0.5],
#                   [0.5,0.5]])
# beta = np.array([[0.5,0.5],
#                   [0.5,0.5]])
# A = np.array([[0.5, 2.0],
#               [0.2, 0.05]])  # Total Factor Productivity

# L = np.array([[50,50],
#               [50,50]])
# K = np.array([[50,50],
#               [50,50]])
# print(production_function(A, L, K, alpha, beta, func='C-D'))

def wage_function(A, L, K, alpha,beta,p, 
                  p_function=production_function,
                  algorithm='marginal_product',
                  share=1):
    
    if algorithm=='marginal_product':
        inc_labor = L + 1
        inc_production = p_function(A, inc_labor, K, alpha, beta)
        Q = p_function(A, L, K, alpha, beta)
        wage = p*(inc_production - Q)
        roi = ((Q*p) - (wage*L))/K
    
    elif algorithm=='share_of_product':
        Q = p_function(A, L, K, alpha, beta)
        wage = share* p * Q/L
        roi = (1-share) * p * Q/K

    elif algorithm=='share_of_marginal_product':
        Q = p_function(A, L, K, alpha, beta)
        inc_labor = L + 1
        inc_production = p_function(A, inc_labor, K, alpha, beta)
        wage = share*p*(inc_production - Q)
        roi = ((Q*p) - (wage*L))/K

    return wage, roi

def demand_function(Y, P):
    D = np.zeros((len(P)))
    for p in range(len(P)):
        D[p] = Y / (len(P) * P[p])

    return D


def plot_gdp(gdp_values, case, country_names=None):
    """
    Plots the GDP of the world and the distribution of GDP of individual countries.

    :param gdp_values: A list of GDP values for each country.
    :param country_names: A list of country names. Defaults to None.
    """
    # Total World GDP
    total_world_gdp = sum(gdp_values)
    
    # Plot Total GDP of the World
    plt.figure(figsize=(10, 5))
    plt.bar('World', total_world_gdp, color='skyblue')
    plt.title('Total GDP of the World')
    plt.ylabel('GDP Value')
    plt.show()

    # Plot Distribution of GDP of Individual Countries
    plt.figure(figsize=(15, 8))
    
    if country_names:
        sns.barplot(x=country_names, y=gdp_values, palette="viridis")
    else:
        sns.barplot(x=list(range(len(gdp_values))), y=gdp_values, palette="viridis")
        
    plt.title('Distribution of GDP of Individual Countries')
    plt.ylabel('GDP Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/{}_vectorised_gdp.png'.format(case))
   

def plot_gdp_distribution(case, gdp_values):
    # Sort GDP values in ascending order
    sorted_gdp = np.sort(gdp_values)
    
    # Create a line plot
    plt.figure(figsize=(10,6))
    # plt.plot(sorted_gdp, marker='o', linestyle='-')
    sns.kdeplot(sorted_gdp, shade=True)

    # Label the axes and the plot
    plt.xlabel('Country Rank by GDP')
    plt.ylabel('GDP')
    plt.title('Distribution of GDP Among Countries')
    plt.grid(True, which="both", ls="--")
    plt.savefig('plots/{}_vectorised_gdp_distribution.png'.format(case))


def innovate(A,K, Q, method='investment_based', beta_lbd=0.01, eta=0.05, beta_diffusion=0.02):
    """
    A function to simulate technological innovation.
    
    Parameters:
    - A: Technology matrix of shape (c, p)
    - production: Current production matrix of shape (c, p)
    - method: Method of innovation: 'investment_based', 'learning_by_doing', or 'diffusion'
    - beta_lbd: Parameter for the learning by doing method
    - eta: Rate of technology progress if innovation is successful
    - beta_diffusion: Parameter to control the rate of technological diffusion
    
    Returns:
    - Updated Technology matrix
    """
    
    if method == 'investment_based':
        
        global_investment = np.sum(K, axis=1)
        K_bar = K / global_investment
        
        theta_in = 1 - np.exp(-beta_lbd * K_bar)
        
        successful_innovation = np.random.rand(*A.shape) < theta_in
        A[successful_innovation] = A[successful_innovation] * (1 + eta)
        
    elif method == 'learning_by_doing':
        
        Q_bar = Q / np.max(Q, axis=1)
        theta_lbd = 1 - np.exp(-beta_lbd * Q_bar)
        
        successful_innovation = np.random.rand(*A.shape) < theta_lbd
        A[successful_innovation] = A[successful_innovation] * (1 + eta)
    
    elif method == 'diffusion':
        global_average_technology = np.mean(A, axis=0)
        
        # Those below the global average have a higher chance of innovating
        tech_below_avg = A < global_average_technology
        
        theta_diffusion = 1 - np.exp(-beta_diffusion * tech_below_avg.astype(float))
        
        successful_innovation = np.random.rand(*A.shape) < theta_diffusion
        A[successful_innovation] = A[successful_innovation] * (1 + eta)
        
    return A


def plot_sorted_production_matrix(case, production_matrix, A_matrix):
    """
    Plots the production matrix and A matrix after sorting rows by their sum 
    and columns by their sum from the production matrix.
    
    production_matrix: 2D array-like structure representing the production of each product by each country.
    A_matrix: 2D array-like structure representing the technological advancement for each product by each country.
    """
    
    # Get the order of rows and columns based on sums of the production matrix
    row_order = np.argsort(np.sum(production_matrix, axis=1))[::-1]
    col_order = np.argsort(np.sum(production_matrix, axis=0))[::-1]
    
    # Sort the matrices
    sorted_production = production_matrix[row_order, :]
    sorted_production = sorted_production[:, col_order]
    
    sorted_A = A_matrix[row_order, :]
    sorted_A = sorted_A[:, col_order]
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(sorted_production, cmap="YlGnBu", ax=ax[0], cbar_kws={'label': 'Production Value'})
    ax[0].set_xlabel('Products (sorted by total production)')
    ax[0].set_ylabel('Countries (sorted by total production)')
    ax[0].set_title('Sorted Production Matrix')
    
    sns.heatmap(sorted_A, cmap="plasma", ax=ax[1], cbar_kws={'label': 'Technology Value'})
    ax[1].set_xlabel('Products (sorted by total production)')
    ax[1].set_ylabel('Countries (sorted by total production)')
    ax[1].set_title('Sorted A Matrix')
    
    plt.tight_layout()
    plt.savefig('plots/{}_vectorised_production.png'.format(case))


import networkx as nx
import matplotlib.pyplot as plt

def visualize_multi_layer_trade_network(trades, n_countries, n_products):
    # Create a multi-layer network
    G = nx.MultiGraph()

    # Add nodes for countries
    for country in range(n_countries):
        G.add_node(country)

    # Add edges for trade relationships in each product layer
    for product in range(n_products):
        for countryA in range(n_countries):
            for countryB in range(n_countries):
                # Ensure you are within bounds of available product indices
                if product < trades.shape[2]:
                    product_trades = trades[countryA, countryB, product, :]
                    total_trade = sum(product_trades)
                    if total_trade > 0:
                        G.add_edge(countryA, countryB, weight=total_trade, product=product)

    # Draw the multi-layer trade network
    pos = nx.spring_layout(G, seed=42)
    edge_colors = [G[u][v][0]['product'] for u, v in G.edges()]
    widths = [G[u][v][0]['weight']*1e-2 for u, v in G.edges()]

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', width=widths, edge_color=edge_colors)

    # # Create an edge_labels dictionary
    # edge_labels = {}
    # for u, v, data in G.edges(data=True):
    #     edge_labels[(u, v, 0)] = f"{data['weight']:.2f}"  # We assume there's only one key for each edge

    # # Add edge labels using edge_labels dictionary
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='red', rotate=False)

    plt.title("Multi-Layer Trade Network")
    plt.show()

import numpy as np

def generate_nested_matrix(rows, cols, nestedness_level=1, max_value=2):
    matrix = np.zeros((rows, cols))

    # Set nestedness_level * min(rows, cols) rows with random values between 0 and max_value in a nested pattern
    nested_rows = int(nestedness_level * min(rows, cols))
    for i in range(nested_rows):
        num_non_zero = min(i + 1, cols)
        non_zero_elements = np.random.uniform(0.1, max_value, size=num_non_zero)
        np.random.shuffle(non_zero_elements)
        matrix[i, :num_non_zero] = non_zero_elements

    # Ensure no row or column has only zeros
    for i in range(rows):
        if np.all(matrix[i, :] == 0):
            matrix[i, np.random.choice(cols)] = np.random.uniform(0.01, max_value)

    for j in range(cols):
        if np.all(matrix[:, j] == 0):
            matrix[np.random.choice(rows), j] = np.random.uniform(0.01, max_value)

    return matrix


def array_to_dataframe(array_3d, index_labels):
    """
    Converts a 3-D array to a DataFrame with the first two indices of the array forming a MultiIndex.

    :param array_3d: 3-D numpy array to be converted.
    :param index_labels: Tuple of lists containing labels for the first and second indices respectively.
    :return: DataFrame with MultiIndex created from the first two dimensions of array_3d.
    """
    # Validate the dimensions of the array and the length of index labels
    if array_3d.ndim != 3 or any(len(labels) != array_3d.shape[i] for i, labels in enumerate(index_labels)):
        raise ValueError("Dimensions of the array and index labels must match.")

    # Generate MultiIndex
    index_tuples = [(level1, level2) for level1 in index_labels[0] for level2 in index_labels[1]]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['country', 'country'])

    # Flatten the 3-D array to 2-D
    reshaped_array = array_3d.reshape(-1, array_3d.shape[2])

    # Create DataFrame
    df = pd.DataFrame(reshaped_array, index=multi_index)

    # stack, unstack to transpose
    df = df.stack().unstack(level=1)

    return df

def regularise(df):
    df2 = df
    n=len(df.index.get_level_values(0).unique())
    m=len(df.index.get_level_values(1).unique())
    denominator=np.ones(m*n)
    for i in range(n):
        for j in range(m):
            ind = m*i + j
            denominator[ind] = df.iloc[ind,i]
            df2.iloc[ind,:] = df.iloc[ind,:]/df.iloc[ind,i]

    return df2

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






