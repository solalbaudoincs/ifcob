# MVP : IFCOG: Information Flow in Crypto Order-Books

Lagged correlations reveal when the past of one asset influences another's future; at the high
frequency level, order-book data are asynchronous, making standard discrete-lag techniques
unsuitable. Here, we study limit order books of multiple cryptocurrencies to uncover cross-crypto
lead-lag signals. The task is to design and backtest a strategy that uses order-book features (e.g.,
bid-ask imbalance, depth shifts) of one coin to predict short-term returns on another, assessing
out-of-sample performance (Sharpe, hit rate) and net profitability after transaction costs.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
  - [Backtesting](#backtesting-backtesting)
  - [Data Preprocessing](#data-preprocessing-preprocessing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview
A modular framework for analyzing information flow in crypto order-books, featuring event-driven backtesting, advanced data preprocessing, and support for custom trading strategies.

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd ifcob
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

### Backtesting (`backtesting/`)
The `backtesting` module evaluates trading strategies using historical order-book data.

![Backtesting Workflow](/images/backtesting_workflow.png)

**Key Features:**
- Event-driven simulation engine
- Support for custom strategies and signals
- Realistic transaction cost modeling
- Performance metrics: Sharpe ratio, hit rate, drawdown, and net P&L
- Modular design for easy extension

**Quick Start:**
```python
from backtesting import Backtester, Strategy

# Define your strategy
class MyStrategy(Strategy):
    def generate_signals(self, data):
        # Implement signal logic
        pass

# Run backtest
bt = Backtester(data, MyStrategy, transaction_costs)
results = bt.run()
print(results.summary())
```

See [`backtesting/README.md`](backtesting/README.md) for detailed documentation.

---

### Data Preprocessing (`preprocessing/`)

The `preprocessing` module provides tools for converting raw cryptocurrency CSV data into a structured format suitable for feature extraction and analysis.

**Key Features:**
- Vectorized preprocessing with configurable block sizes
- Automatic duplicate removal
- Conversion to wide format for analysis
- Support for both single file and batch processing
- Importable module with clean API

**Quick Start:**
```python
from preprocessing import preprocess_crypto_data, preprocess_data_folder

# Process single file
df = preprocess_crypto_data('data/raw/DATA_0/XBT_EUR.csv', 'data/preprocessed/XBT_EUR.parquet', 'XBT')

# Process entire folder
results = preprocess_data_folder('data/raw/DATA_0', 'data/preprocessed/DATA_0')
```

**Command Line Usage:**
```bash
# Process all files in a folder
python preprocessing/preprocess_script.py -i data/raw/DATA_0 -o data/preprocessed/DATA_0

# Process specific coins
python preprocessing/preprocess_script.py -i data/raw/DATA_0 -o data/preprocessed/DATA_0 --coins XBT ETH
```

See [`preprocessing/README.md`](preprocessing/README.md) for detailed documentation.

---

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License
This project is licensed under the terms of the LICENSE file in this repository.

## Contact
For questions or collaboration, please contact the project maintainer.
