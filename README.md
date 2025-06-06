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
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
  - [Backtesting](#backtesting-backtesting)
  - [Data Preprocessing](#data-preprocessing-preprocessing)
  - [Automated Backtesting](#automated-backtesting-scripts)
- [Strategy Development](#strategy-development)
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

## Quick Start

### Running a Backtest
Use the automated backtesting script to quickly test strategies:

```bash
# Run a single strategy with data index 1
python scripts/run_backtest.py --strategy TFCumulativeReturnStrategy --data-index 1

# Run multiple strategies with different data versions
python scripts/run_backtest.py --strategy Mateo2StartStrategy --strategy RFPredAllSignedStratMateoCheating --data-index 2

# Save detailed results
python scripts/run_backtest.py --strategy MyStrategy --data-index 1 --output-dir results --save-trades --save-portfolio
```

### List Available Strategies
```bash
python scripts/run_backtest.py --list-strategies
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

### Automated Backtesting (`scripts/`)

The `scripts/run_backtest.py` provides a comprehensive command-line interface for running backtests with automatic strategy discovery and data versioning.

**Key Features:**
- **Automatic Strategy Discovery**: Dynamically finds all strategies in the `strategies/` folder
- **Data Index Management**: Mandatory `--data-index` parameter ensures explicit data versioning
- **Automatic Feature Generation**: Checks for and generates precomputed features when needed
- **Precomputed Features Support**: Automatic detection and handling of strategies that load precomputed features
- **Flexible Configuration**: Support for custom data sources, fees, and strategy parameters
- **Comprehensive Results**: Detailed output with performance metrics, trade logs, and portfolio evolution
- **Profiling Support**: Built-in performance profiling for optimization

#### Data Index System

The project uses a data index system to manage different versions of preprocessed data:

```
data/features/
├── DATA_1/          # First dataset version
│   ├── XBT_EUR.parquet
│   └── ETH_EUR.parquet
├── DATA_2/          # Second dataset version
│   ├── XBT_EUR.parquet
│   └── ETH_EUR.parquet
└── ...
```

**Strategy Compatibility:**
- ✅ **Supports `--data-index`**: Strategies with `data_index` parameter automatically use the specified dataset
- ⚠️ **Hardcoded Paths**: Strategies without `data_index` parameter may use hardcoded paths and need manual updates
- ℹ️ **No Precomputed Features**: Strategies that don't load external data work with any dataset

#### Automatic Feature Generation

The backtesting script automatically handles precomputed features:

1. **Detection**: Analyzes strategy code to detect if precomputed features are needed
2. **Verification**: Checks if feature files exist for the specified data index
3. **Generation**: Automatically generates missing features using `scripts/generate_features.py`
4. **Interactive Mode**: Prompts user before generating features (unless `--auto-generate-features` is used)
5. **Error Handling**: Provides clear error messages and manual generation commands if automatic generation fails

**Feature Generation Options:**
- `--auto-generate-features`: Generate features automatically without prompting (non-interactive mode)
- `--force-feature-check`: Check for features even for strategies that may not need them
- `--ignore-feature-errors`: Continue with backtesting even if feature generation fails

#### Usage Examples

**Basic Usage:**
```bash
# Required: Always specify data index
python scripts/run_backtest.py --strategy MyStrategy --data-index 1

# Multiple strategies
python scripts/run_backtest.py --strategy Strategy1 --strategy Strategy2 --data-index 2
```

**Automatic Feature Generation:**
```bash
# Non-interactive mode (auto-generates features without prompting)
python scripts/run_backtest.py --strategy Mateo2StartStrategy --data-index 2 --auto-generate-features

# Force feature check for all strategies
python scripts/run_backtest.py --strategy SimpleStrategy --data-index 1 --force-feature-check

# Continue even if feature generation fails
python scripts/run_backtest.py --strategy MyStrategy --data-index 1 --ignore-feature-errors
```

**Advanced Configuration:**
```bash
# Custom parameters
python scripts/run_backtest.py \
  --strategy RFPredAllSignedStratMateo \
  --data-index 1 \
  --window-size 10 \
  --threshold 0.05 \
  --initial-capital 500000 \
  --fee-percentage 0.15

# Custom data sources (override defaults)
python scripts/run_backtest.py \
  --strategy MyStrategy \
  --data-index 1 \
  --data-sources XBT:data/custom/XBT.parquet ETH:data/custom/ETH.parquet

# Save detailed results
python scripts/run_backtest.py \
  --strategy MyStrategy \
  --data-index 1 \
  --output-dir backtest_results \
  --save-trades \
  --save-portfolio
```

**Performance Analysis:**
```bash
# Profile execution
python scripts/run_backtest.py --strategy MyStrategy --data-index 1 --profile

# Custom split ratio for calibration/validation
python scripts/run_backtest.py --strategy MyStrategy --data-index 1 --split-ratio 0.8
```

#### Workflow Example

When running a strategy that requires precomputed features:

```bash
$ python scripts/run_backtest.py --strategy Mateo2StartStrategy --data-index 2

============================================================
BACKTESTING ENGINE
============================================================
Using default data sources for DATA_2: [('XBT', 'data/features/DATA_2/XBT_EUR.parquet'), ('ETH', 'data/features/DATA_2/ETH_EUR.parquet')]
Data index: 2
Loaded strategy: Mateo2StartStrategy
  → Uses precomputed features with data_index=2

============================================================
CHECKING PRECOMPUTED FEATURES
============================================================
Strategy 'Mateo2StartStrategy' requires precomputed features
⚠ Precomputed features for DATA_2 not found
Generate precomputed features for DATA_2? [Y/n]: y
Generating precomputed features for DATA_2...
  Generating features for XBT...
  Generating features for ETH...
✓ Successfully generated precomputed features for DATA_2

Trading fees: 0.1%
Dataloader initialized successfully
...
```

#### Output Files

When using `--output-dir`, the script generates:

- **`backtest_summary_TIMESTAMP.csv`**: High-level performance metrics
- **`backtest_results_TIMESTAMP.json`**: Detailed results in JSON format
- **`trades_details_STRATEGY_PHASE_TIMESTAMP.csv`**: Individual trade records (with `--save-trades`)
- **`portfolio_evolution_STRATEGY_PHASE_TIMESTAMP.csv`**: Portfolio value over time (with `--save-portfolio`)

---

## Strategy Development

### Creating Data-Index Compatible Strategies

To make your strategy compatible with the data index system, add a `data_index` parameter:

```python
from backtesting.strategy import Strategy
import pandas as pd
import joblib

class MyStrategy(Strategy):
    def __init__(self, data_index=1, window_size=5):
        super().__init__()
        # Use data_index for loading precomputed features
        self.features_df = pd.read_parquet(f"data/features/DATA_{data_index}/XBT_EUR.parquet")
        self.model = joblib.load(f"predictors/my_model_{window_size}ms.joblib")
    
    def get_action(self, data, current_portfolio, fees_graph):
        # Your strategy logic here
        pass
```

### Strategy Discovery

The automated script discovers strategies by:
1. Scanning all `.py` files in the `strategies/` folder
2. Finding classes that inherit from `Strategy`
3. Checking for `data_index` parameter support
4. Detecting precomputed feature usage patterns

### Best Practices

1. **Always use `data_index`** parameter for loading precomputed features
2. **Test with multiple data versions** to ensure robustness
3. **Use descriptive strategy names** and documentation
4. **Handle missing features gracefully** in your strategy logic
5. **Profile your strategies** to optimize execution time
6. **Use `--auto-generate-features`** for automated CI/CD pipelines

### Feature Generation Integration

When developing strategies that use precomputed features:

1. **Dependency Check**: The backtesting script automatically detects if your strategy loads precomputed features
2. **Automatic Generation**: Missing features are generated automatically using the `generate_features.py` script
3. **Data Versioning**: Features are generated for the specific `data_index` you specify
4. **Error Recovery**: If automatic generation fails, manual commands are provided

```python
# Example strategy with automatic feature dependency detection
class MyMLStrategy(Strategy):
    def __init__(self, data_index=1):
        super().__init__()
        # This will be automatically detected as requiring precomputed features
        self.model = joblib.load("predictors/my_model.joblib")
        self.features = pd.read_parquet(f"data/features/DATA_{data_index}/XBT_EUR.parquet")
```

---

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License
This project is licensed under the terms of the LICENSE file in this repository.

## Contact
For questions or collaboration, please contact the project maintainer.
