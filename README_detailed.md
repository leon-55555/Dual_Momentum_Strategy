# Dual Momentum Crypto Strategy

This repository presents an independent quantitative research project based on dual momentum investing principles applied to a curated universe of liquid cryptocurrencies. The strategy combines ideas from empirical asset pricing, momentum literature, and robust Python-based backtesting to construct and evaluate a daily portfolio. The work was carried out independently and designed to showcase proficiency in research design, signal engineering, and performance evaluation, as expected in quantitative research roles.

## Strategy Overview

Dual momentum is a systematic investing approach that blends two well-documented effects:

- **Relative Momentum**: This concept ranks assets based on past performance and selects those with the highest returns over a given period. As shown by Jegadeesh and Titman (1993), past winners tend to continue outperforming past losers over intermediate time horizons.

- **Absolute Momentum**: This component filters positions by checking whether each asset's recent return is positive or negative. This concept, emphasised in Antonacci's "Dual Momentum Investing" (2015), adds a defensive layer to avoid exposure during bear phases or downtrends.

In this project, the dual momentum strategy is applied to cryptocurrencies on a **daily frequency**. On each day:

1. A z-score is computed for each coin based on its recent return relative to its rolling mean and volatility.
2. The top-n\_ assets are selected by z-score rank.
3. These assets are included in the portfolio only if their absolute return is positive.
4. Positions are equally weighted (or weighted by relative z-score in the upgraded version).
5. If no assets pass the absolute filter, the portfolio holds cash.

## Signal Construction

The strategy uses a rolling z-score of returns to compare assets. This method standardises momentum signals across assets with different volatilities:

The z-score is given by $z = \frac{r - \mu_l}{\sigma_l}$.
Where:

- \( r \) is the return over a short lookback window (`l_1`)
- \( \mu_l \), \( \sigma_l \) are the rolling mean and standard deviation over a separate window (`l_2`)

This signal provides a more normalised view of momentum across assets, reducing bias toward highly volatile or trending coins.

## Parameters

The primary hyperparameters controlling the strategy are:

- `l_1`: The return lookback period used to calculate recent performance.
- `l_2`: The rolling window used to compute the z-score mean and volatility.
- `top_n`: The maximum number of assets to include in the portfolio on any given day.
- `fees`: Transaction cost model. In this implementation, a fixed rate (e.g., 0.2%) is applied to the absolute weight change of each asset.

The combination of two lookbacks (`l_1` and `l_2`) introduces flexibility in signal construction, allowing for a distinction between signal formation and signal normalisation.

## Performance Metrics

The backtest computes comprehensive performance statistics including:

- **Annualized Return** and **Volatility**
- **Sharpe Ratio** and **Sortino Ratio**
- **Max Drawdown** and **Average Drawdown**
- **Winning Rate** (% of days with positive return)
- **Total Trades** and **Average Holding Period**

These metrics help evaluate both return potential and risk characteristics, particularly during the volatile periods common in cryptocurrency markets.

## Optimisation and Robustness Testing

To prevent overfitting, the project includes:

- **Random search** over a parameter grid for `l_1`, `l_2`, and `top_n`.
- **Sensitivity analysis**, showing how the Sharpe ratio responds to individual parameters.
- **Heatmaps** of Sharpe ratio across parameter combinations.
- A **validation test** on a separate period to assess generalisation.

Random search is preferred over grid search due to its efficiency in higher-dimensional parameter spaces, as noted by Bergstra and Bengio (2012).

## Visual Output

The strategy includes plotting functions for:

- Cumulative returns of the strategy vs individual assets
- Drawdowns over time
- Line plots for parameter sensitivity
- 2D heatmaps of Sharpe ratio across combinations of lookback and portfolio size

These visual tools provide intuitive insights into both strategy performance and parameter robustness.

## File Structure

```
dual-momentum-strategy/
├── dual_momentum_strategy.py        # All functions for strategy, evaluation, optimisation, and plotting
├── reports/
│   └── Presentation_Dual_Momentum_Strategy.pdf
├── requirements.txt
└── README.md
```

## Python Dependencies

This project uses the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- tqdm

For a comprehensive guide on full environment setup and package installation, please take a look at the [INSTALLATION.md](./INSTALLATION.md) document.

## References

- Gary Antonacci (2015). _Dual Momentum Investing: An Innovative Strategy for Higher Returns with Lower Risk_. McGraw-Hill.
- Narasimhan Jegadeesh and Sheridan Titman (1993). _Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency_. Journal of Finance.
- Bergstra & Bengio (2012). _Random Search for Hyper-Parameter Optimisation_. Journal of Machine Learning Research.
- Additional practical guidance and examples were drawn from sources such as QuantInsti, QuantStart, and Investopedia.

## Author

Leon Scheinfeld
Independent Quantitative Researcher  
April 2025  
https://github.com/leon-55555

## Notes and Limitations

This strategy was backtested on eight surviving cryptocurrencies, which introduces potential **survivor bias**. The test period may also benefit from regime-specific behaviour (e.g., recovery from market crashes). Expanding the asset universe, including long-short extensions, and applying volatility targeting are possible next steps for future improvement.
