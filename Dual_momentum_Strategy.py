import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import seaborn as sns
import os

# ========== DATA LOADING ==========
def load_prices_from_long_csv(filepath):
    """
    Loads a long-format CSV of price data (columns: date, symbol, value)
    and pivots it into wide format for backtesting.

    Parameters:
        filepath (str): Path to the CSV (e.g., close.csv)

    Returns:
        pd.DataFrame: Pivoted price DataFrame [date x asset]
    """
    df_raw = pd.read_csv(filepath)
    
    # Ensure columns: date, symbol, value (auto-detect if needed)
    if df_raw.shape[1] > 3:
        raise ValueError("CSV must have 3 columns: date, symbol, value (e.g., close price).")

    df_pivot = df_raw.pivot(index='date', columns='symbol', values=df_raw.columns[-1])
    df_pivot.index = pd.to_datetime(df_pivot.index)
    df_pivot = df_pivot.sort_index()
    
    return df_pivot

def get_price_and_return_dfs(folder="~/Downloads/Strategies/Data_val_csv", file="close.csv"):
    """
    Loads wide-format close price data (date x coin columns) and computes returns.

    Parameters:
        folder (str): Folder path
        file (str): File name (e.g. 'close.csv')

    Returns:
        df_prices (DataFrame): Wide-format prices [date x asset]
        df_returns (DataFrame): Daily returns
    """
    path = os.path.expanduser(os.path.join(folder, file))
    
    # Read wide-format CSV (first col = date, others = asset columns)
    df_prices = pd.read_csv(path)
    
    # Clean up: set date as index and convert to datetime
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    df_prices = df_prices.set_index('date').sort_index()
    
    # Ensure numeric columns (convert error cells to NaN)
    df_prices = df_prices.apply(pd.to_numeric, errors='coerce')

    # Compute daily % returns
    df_returns = df_prices.pct_change().fillna(0)

    return df_prices, df_returns
# ========== CORE STRATEGY FUNCTIONS ==========

def get_weight(params, df_prices):
    """
    Generate equal-weight portfolio signals based on dual momentum.

    This function selects the top_n assets using their z-score of returns 
    (relative momentum) and includes only those with positive recent return 
    (absolute momentum). Assets that pass both filters are equally weighted.

    Parameters:
        params (dict): Contains:
            - l_1: lookback for return computation
            - l_2: lookback for rolling mean and std (used in z-score)
            - top_n: number of assets to select
        df_prices (DataFrame): Price data (indexed by date, columns as assets)

    Returns:
        DataFrame: Portfolio weights (binary equal weights per day)
    """
    l_1 = params['l_1']
    l_2 = params['l_2']
    top_n = params['top_n']

    # Compute return over the l_1 period
    df_returns = df_prices.pct_change(l_1)

    # Compute rolling mean and std dev over l_2 days (for z-score)
    df_rolling_mean = df_returns.rolling(window=l_2, min_periods=1).mean()
    df_rolling_std = df_returns.rolling(window=l_2, min_periods=1).std()

    # Z-score standardization
    df_z = (df_returns - df_rolling_mean) / df_rolling_std

    # Rank assets by momentum signal (higher is better)
    df_rank = df_z.rank(axis=1, ascending=False)

    # Select top_n assets and only those with positive return
    df_selected = df_rank[df_rank <= top_n]
    df_filtered = df_selected.where(df_returns > 0)

    # Normalize: equal weights across selected assets
    df_weight = df_filtered.notnull().div(df_filtered.count(axis=1), axis=0)
    return df_weight.fillna(0)

def get_weight_upg(params, df_prices):
    """
    Generate z-score-weighted portfolio from selected top_n assets.

    Instead of assigning equal weight, the z-score magnitude determines 
    the relative weighting among selected assets.

    Parameters:
        params (dict): Contains l1, l2, top_n
        df_prices (DataFrame): Daily price data

    Returns:
        DataFrame: Weighted portfolio based on standardized momentum
    """
    l_1 = params['l1']
    l_2 = params['l2']
    top_n = params['top_n']

    df_returns = df_prices.pct_change(l_1)
    df_rolling_mean = df_returns.rolling(window=l_2, min_periods=1).mean()
    df_rolling_std = df_returns.rolling(window=l_2, min_periods=1).std()

    df_z = (df_returns - df_rolling_mean) / df_rolling_std
    df_rank = df_z.rank(axis=1, ascending=False)

    # Keep z-scores only for top_n assets each day
    df_filtered = df_z.where(df_rank <= top_n)

    # Normalize z-scores to sum to 1 (per day)
    df_weight = df_filtered.div(df_filtered.sum(axis=1), axis=0)
    return df_weight.fillna(0)

# ========== COST MODELING ==========

def calculate_fees(df_weights, params):
    """
    Estimate transaction fees as proportional to weight change per asset.

    Parameters:
        df_weights (DataFrame): Daily portfolio weights
        params (dict): Must include 'fees' (e.g. 0.002 for 0.2%)

    Returns:
        Series: Daily total portfolio fee
    """
    fees = params['fees']
    df_weight_change = df_weights.diff().abs()
    df_fees = df_weight_change * fees
    return df_fees.sum(axis=1).fillna(0)

# ========== STRATEGY EVALUATION ==========

def strat_eval(df_returns, df_weights, df_fees):
    """
    Evaluate strategy performance with and without fees.

    Computes key metrics used in quant research: Sharpe, Sortino, 
    annualized return/volatility, max drawdown, and trade diagnostics.

    Parameters:
        df_returns (DataFrame): Daily asset returns
        df_weights (DataFrame): Daily portfolio weights
        df_fees (Series): Transaction costs per day

    Returns:
        for_the_plots (dict): Equity curves and drawdowns
        performance_metrics (dict): Summary statistics
    """
    # Raw strategy return
    strat_return = (df_weights.shift(1) * df_returns).sum(axis=1).fillna(0)

    # After accounting for fees
    net_return = strat_return - df_fees.shift(1)

    # Cumulative performance (gross and net)
    equity_curve = (1 + strat_return).cumprod()
    equity_curve_fees = (1 + net_return).cumprod()

    # Drawdowns
    drawdowns = (equity_curve_fees / equity_curve_fees.cummax()) - 1
    max_drawdown = drawdowns.min(skipna=True) * 100

    # Risk-adjusted metrics
    sharpe = net_return.mean() / net_return.std() * np.sqrt(365)
    sortino = net_return.mean() / net_return[net_return < 0].std() * np.sqrt(365)
    ann_return = equity_curve_fees.iloc[-1]**(365 / len(net_return)) - 1
    ann_vol = net_return.std() * np.sqrt(365)
    win_rate = (net_return > 0).mean() * 100

    # Trading behavior: detect reallocation events
    rebalanced = (df_weights.diff().abs().sum(axis=1) != 0)
    total_trades = rebalanced.sum()
    strategy_ids = (~rebalanced).cumsum()
    avg_hold = strategy_ids.value_counts().mean()

    for_the_plots = {
        'equity_curve': equity_curve,
        'equity_curve_fees': equity_curve_fees,
        'drawdowns_fees': drawdowns
    }

    performance_metrics = {
        'max_drawdown_fees': max_drawdown,
        'sharpe_fees': sharpe,
        'sortino_ratio': sortino,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'winning_rate': win_rate,
        'total_trades': total_trades,
        'avg_strategy_holding': avg_hold
    }

    return for_the_plots, performance_metrics

def random_search(df_prices, df_weights, df_returns):
    """
    Perform random search to find optimal strategy parameters (l_1, l_2, top_n).

    This method randomly samples combinations of hyperparameters and evaluates 
    the strategy on each set using Sharpe ratio as the selection criterion.

    Parameters:
        df_prices (DataFrame): Asset prices
        df_weights (DataFrame): Initial dummy weights (not used directly here)
        df_returns (DataFrame): Daily asset returns

    Returns:
        DataFrame: Top 5 parameter combinations sorted by Sharpe ratio
    """
    results = []
    n_trials = 500  # Number of random combinations to test

    for _ in range(n_trials):
        # Randomly generate parameters
        l_1  = random.randint(10, 50)
        l_2 = random.randint(10, 50)
        top_n = random.randint(1, 5)

        params = {'l_1': l_1, 'l_2': l_2, 'top_n': top_n, 'fees': 0.002}

        df_weights = get_weight(params, df_prices)
        df_fees = calculate_fees(df_weights, params)
        _, metrics = strat_eval(df_returns, df_weights, df_fees)

        # Store tested parameters with results
        metrics['l_1'] = l_1
        metrics['l_2'] = l_2
        metrics['top_n'] = top_n
        results.append(metrics)

    df_results = pd.DataFrame(results)
    return df_results.sort_values(by='sharpe_fees', ascending=False).head(5)

def batch_sensitivity_analysis_l1(params, df_prices, df_returns, l_1_range):
    """
    Analyze sensitivity of Sharpe ratio to different l_1 values.

    Parameters:
        params (dict): Base parameters to copy and override 'l_1'
        df_prices (DataFrame): Price data
        df_returns (DataFrame): Return data
        l_1_range (list): Values of l_1 to test

    Returns:
        DataFrame: Sharpe ratio for each l_1 value
    """
    results = []
    for val in l_1_range:
        params_c = params.copy()
        params_c['l_1'] = val

        df_weights = get_weight(params_c, df_prices)
        df_fees = calculate_fees(df_weights, params_c)
        _, performance_metrics = strat_eval(df_returns, df_weights, df_fees)

        results.append({'l_1': val, 'sharpe_ratio': performance_metrics['sharpe_fees']})

    return pd.DataFrame(results).set_index('l_1')

def batch_sensitivity_analysis_l2(params, df_prices, df_returns, l_2_range):
    """
    Same as above but tests sensitivity to the z-score window (l_2).
    """
    results = []
    for val in l_2_range:
        params_c = params.copy()
        params_c['l_2'] = val

        df_weights = get_weight(params_c, df_prices)
        df_fees = calculate_fees(df_weights, params_c)
        _, performance_metrics = strat_eval(df_returns, df_weights, df_fees)

        results.append({'l_2': val, 'sharpe_ratio': performance_metrics['sharpe_fees']})

    return pd.DataFrame(results).set_index('l_2')

def batch_sensitivity_analysis_topn(params, df_prices, df_returns, top_n_range):
    """
    Evaluate Sharpe ratio as top_n (portfolio size) changes.
    """
    results = []
    for val in top_n_range:
        params_c = params.copy()
        params_c['top_n'] = val

        df_weights = get_weight(params_c, df_prices)
        df_fees = calculate_fees(df_weights, params_c)
        _, performance_metrics = strat_eval(df_returns, df_weights, df_fees)

        results.append({'top_n': val, 'sharpe_ratio': performance_metrics['sharpe_fees']})

    return pd.DataFrame(results).set_index('top_n')

def sharpe_heatmap(df_prices, df_returns, l_2_range=range(10, 51, 5), top_n_range_h=range(1, 6), fees=0.002, l_1=14):
    """
    Plot a heatmap of Sharpe ratios as l_2 and top_n vary (holding l_1 constant).

    Returns:
        DataFrame: Grid of Sharpe ratios for each combination.
    """
    heatmap_data = []

    for l_2 in l_2_range:
        for top_n in top_n_range_h:
            params = {'l_1': l_1, 'l_2': l_2, 'top_n': top_n, 'fees': fees}
            df_weights = get_weight(params, df_prices)
            df_fees = calculate_fees(df_weights, params)
            _, performance_metrics = strat_eval(df_returns, df_weights, df_fees)

            heatmap_data.append({
                'l_1': l_1,
                'l_2': l_2,
                'top_n': top_n,
                'sharpe_ratio': performance_metrics['sharpe_fees']
            })

    df_heatmap = pd.DataFrame(heatmap_data)
    heatmap_matrix = df_heatmap.pivot(index='top_n', columns='l_2', values='sharpe_ratio')

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, cbar_kws={'label': 'Sharpe Ratio'})
    plt.title('Sharpe Ratio Sensitivity to l_2 and Top N')
    plt.xlabel('l_2 (Z-score Window)')
    plt.ylabel('Top N Assets')
    plt.tight_layout()
    plt.show()

    return df_heatmap

def sharpe_heatmap_l1_l2(df_prices, df_returns, l_1_range=range(10, 51, 5), l_2_range=range(10, 51, 5), top_n=3, fees=0.002):
    """
    Heatmap showing Sharpe ratio across combinations of l_1 and l_2.
    """
    heatmap_data = []

    for l_1 in l_1_range:
        for l_2 in l_2_range:
            params = {'l_1': l_1, 'l_2': l_2, 'top_n': top_n, 'fees': fees}
            df_weights = get_weight(params, df_prices)
            df_fees = calculate_fees(df_weights, params)
            _, performance_metrics = strat_eval(df_returns, df_weights, df_fees)

            heatmap_data.append({
                'l_1': l_1,
                'l_2': l_2,
                'top_n': top_n,
                'sharpe_ratio': performance_metrics['sharpe_fees']
            })

    df_heatmap = pd.DataFrame(heatmap_data)
    heatmap_matrix = df_heatmap.pivot(index='l_1', columns='l_2', values='sharpe_ratio')

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, cbar_kws={'label': 'Sharpe Ratio'})
    plt.title(f'Sharpe Ratio Sensitivity to l_1 and l_2 (top_n={top_n})')
    plt.xlabel('l_2 (Z-score Window)')
    plt.ylabel('l_1 (Return Lookback)')
    plt.tight_layout()
    plt.show()

    return df_heatmap

if __name__ == "__main__":
    # Load close prices and compute returns from long-format CSV
    df_prices, df_returns = get_price_and_return_dfs(
        folder="~/Downloads/Strategies/Data_val_csv",
        file="close.csv"
    )

    # Base parameter configuration
    params = {
        'l_1': 25,     # Lookback period for raw returns (absolute momentum)
        'l_2': 19,     # Lookback period for z-score normalization (relative momentum)
        'top_n': 3,    # Max number of assets selected daily
        'fees': 0.002  # Proportional transaction cost (0.2%)
    }

    # Parameter ranges for sensitivity analysis
    l_1_range = list(range(1, 51))
    l_2_range = list(range(1, 51))
    top_n_range = list(range(1, 6))

    # Generate portfolio weights using strategy logic
    df_weights = get_weight(params, df_prices)

    # Compute transaction costs from weight changes
    df_fees = calculate_fees(df_weights, params)

    # Evaluate strategy returns, drawdowns, Sharpe, Sortino, etc.
    for_the_plots, performance_metrics = strat_eval(df_returns, df_weights, df_fees)

    # -------------------------
    # Optimization Experiments
    # -------------------------

    # Random search: find best-performing hyperparameter combinations
    df_random_search = random_search(df_prices, df_weights, df_returns)
    print("Top 5 Random Search Results:\n", df_random_search)

    # -------------------------
    # Sensitivity Analyses (1D)
    # -------------------------

    # Sharpe ratio as function of l_1 (return window)
    df_sharpe_l_1 = batch_sensitivity_analysis_l1(params, df_prices, df_returns, l_1_range)
    df_sharpe_l_1.plot(title='Sharpe Sensitivity to l_1', legend=False)

    # Sharpe ratio as function of l_2 (normalization window)
    df_sharpe_l_2 = batch_sensitivity_analysis_l2(params, df_prices, df_returns, l_2_range)
    df_sharpe_l_2.plot(title='Sharpe Sensitivity to l_2', legend=False)

    # Sharpe ratio as function of top_n (number of selected assets)
    df_sharpe_n = batch_sensitivity_analysis_topn(params, df_prices, df_returns, top_n_range)
    df_sharpe_n.plot(title='Sharpe Sensitivity to Top N', legend=False)

    # -------------------------
    # Sensitivity Analyses (2D Heatmaps)
    # -------------------------

    # Heatmap: l_2 vs top_n, fixed l_1
    df_heatmap = sharpe_heatmap(
        df_prices, df_returns,
        l_2_range=range(1, 51, 2),
        top_n_range_h=range(1, 6),
        fees=params['fees'],
        l_1=params['l_1']
    )

    # Heatmap: l_1 vs l_2, fixed top_n
    df_heatmap_l1_l2 = sharpe_heatmap_l1_l2(
        df_prices, df_returns,
        l_1_range=range(10, 51, 5),
        l_2_range=range(10, 51, 5),
        top_n=params['top_n'],
        fees=params['fees']
    )
