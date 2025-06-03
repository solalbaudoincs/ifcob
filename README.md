# MVP : Information Flow in Crypto Order-Books

Lagged correlations reveal when the past of one asset influences another’s future; at the high
frequency level, order-book data are asynchronous, making standard discrete-lag techniques
unsuitable. Here, we study limit order books of multiple cryptocurrencies to uncover cross-crypto
lead-lag signals. The task is to design and backtest a strategy that uses order-book features (e.g.,
bid-ask imbalance, depth shifts) of one coin to predict short-term returns on another, assessing
out-of-sample performance (Sharpe, hit rate) and net profitability after transaction costs.

