# Random Baseline Recall Confidence Intervals

## 📚 **Mathematical Foundation**

### **Problem Statement**
Given a target series with signals `{-1, 0, 1}` and their proportions `{p₋₁, p₀, p₁}`, compute the confidence intervals for recall under a random baseline predictor that follows the target distribution.

### **Random Baseline Model**
A random predictor that outputs signal `s` with probability `pₛ` (matching target proportions):
- `P(predict = -1) = p₋₁`
- `P(predict = 0) = p₀` 
- `P(predict = 1) = p₁`

### **Recall Under Random Baseline**

For signal `s`, recall is defined as:
```
Recall(s) = TP(s) / Actual_Positives(s)
```

Where:
- `TP(s)` = True Positives for signal `s`
- `Actual_Positives(s)` = Total occurrences of signal `s` in targets

### **Key Insight: Expected Recall = Signal Proportion**

Under random prediction:
```
E[Recall(s)] = E[TP(s)] / Actual_Positives(s)
             = Actual_Positives(s) × P(predict = s) / Actual_Positives(s)
             = P(predict = s)
             = pₛ
```

**Therefore: Expected recall for signal `s` equals its proportion in the target!**

### **Confidence Interval Calculation**

Since `TP(s) ~ Binomial(Actual_Positives(s), pₛ)`, we use the Wilson score method from `proportion_confint()`:

```python
lower, upper = proportion_confint(
    count=expected_tp,        # = Actual_Positives(s) × pₛ
    nobs=actual_positives,    # = Actual_Positives(s)
    alpha=1 - confidence,
    method='wilson'
)
```

## 🔧 **Implementation Details**

### **Core Functions**

1. **`recall_interval_random_baseline(targets, signal, confidence)`**
   - Computes CI for a single signal
   - Returns: `(lower_bound, upper_bound)`

2. **`compute_all_recall_intervals_random_baseline(targets, confidence)`**
   - Computes CIs for all signals present in targets
   - Returns: Dictionary with detailed results

3. **`theoretical_recall_distribution(targets, signal)`**
   - Provides theoretical statistics (mean, variance, std)
   - Useful for validation and understanding

### **Example Usage**

```python
import pandas as pd
from perf_analysis import compute_all_recall_intervals_random_baseline

# Your target data
targets = pd.Series([-1, -1, 0, 0, 0, 1, 1, -1, 0, 1])

# Compute confidence intervals
results = compute_all_recall_intervals_random_baseline(targets, confidence=0.95)

# Access results
for signal in [-1, 0, 1]:
    if signal in results:
        r = results[signal]
        print(f"Signal {signal}: Expected recall = {r['expected_recall']:.3f}")
        print(f"  95% CI: [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]")
```

## 📊 **Interpretation Guide**

### **What the Results Tell You**

1. **Expected Recall = Signal Proportion**
   - Signal appearing 30% of the time → Expected recall = 0.300
   - This is the baseline performance for random prediction

2. **Confidence Interval Width**
   - Wider CIs for rarer signals (smaller sample sizes)
   - Narrower CIs for frequent signals (larger sample sizes)

3. **Performance Evaluation**
   - Your model's recall should significantly exceed the upper CI bound
   - If recall falls within the CI, it's not better than random chance

### **Practical Applications**

1. **Model Validation**
   ```python
   # Check if your model beats random baseline
   model_recall = 0.65
   random_ci_upper = results[1]['ci_upper']  # For signal 1
   
   if model_recall > random_ci_upper:
       print("✅ Model significantly beats random baseline!")
   else:
       print("❌ Model not significantly better than random")
   ```

2. **Statistical Significance Testing**
   ```python
   # Compare multiple models against baseline
   models = {'ModelA': 0.45, 'ModelB': 0.52, 'ModelC': 0.38}
   baseline_upper = results[1]['ci_upper']
   
   for name, recall in models.items():
       status = "✅ Significant" if recall > baseline_upper else "❌ Not significant"
       print(f"{name}: {recall:.3f} | {status}")
   ```

3. **Feature Engineering Validation**
   - Use random baseline CIs to validate new features
   - Ensure improvements are statistically meaningful
   - Avoid overfitting to noise

## 🎨 **Visualization Capabilities**

The performance analysis toolkit now includes comprehensive visualization functions to help understand and communicate results:

### **1. Random Baseline Analysis (`plot_recall_confidence_intervals`)**
- **4-panel dashboard** showing:
  - Expected recalls with confidence intervals (bar chart with error bars)
  - Signal distribution (pie chart)
  - Confidence interval widths (bar chart)
  - Summary statistics panel
- **Key Features**:
  - Color-coded by signal type (red=-1, gray=0, green=1)
  - Value labels on all charts
  - Comprehensive statistical summary

### **2. Prediction Performance Analysis (`plot_prediction_performance`)**
- **4-panel comprehensive analysis**:
  - Actual vs Expected recall comparison with significance markers (★ for significant)
  - Performance improvement over random baseline
  - Statistical significance summary with detailed results
  - Prediction vs target distribution comparison
- **Significance Indicators**:
  - ★ Gold star: Statistically significant performance
  - ○ Gray circle: Not significant
  - Color coding: Green for positive improvement, red for negative

### **3. Theoretical Validation (`plot_theoretical_validation`)**
- **Monte Carlo simulation validation** of analytical confidence intervals
- **3-panel validation** (one per signal):
  - Histogram of simulated recall values
  - Analytical expected value (red dashed line)
  - Analytical confidence intervals (orange dotted lines) 
  - Simulation mean (blue solid line)
- **Validation metrics** showing agreement between analytical and simulation results

### **Usage Examples**

```python
from perf_analysis import (
    plot_recall_confidence_intervals,
    plot_prediction_performance, 
    plot_theoretical_validation
)

# Visualize random baseline confidence intervals
fig1 = plot_recall_confidence_intervals(targets, confidence=0.95)

# Analyze prediction performance against baseline
fig2 = plot_prediction_performance(predictions, targets, confidence=0.95)

# Validate analytical approach with simulation
fig3 = plot_theoretical_validation(targets, confidence=0.95, n_simulations=10000)
```

### **Demonstration Script**
Run `demo_visualizations.py` for comprehensive examples showing:
- Random baseline analysis
- Poor vs good model comparison
- Theoretical validation
- Statistical significance interpretation

## ⚠️ **Important Notes**

### **Assumptions**
1. **Independent Predictions**: Each prediction is independent
2. **Known Distribution**: Random baseline follows target proportions
3. **Sufficient Sample Size**: Wilson method works best with adequate samples

### **Limitations**
1. **No Temporal Dependencies**: Assumes predictions are i.i.d.
2. **Single Metric**: Only addresses recall, not precision or F1
3. **No Class Imbalance Correction**: Raw proportions used as probabilities

### **When to Use**
- ✅ Establishing statistical baselines
- ✅ Model comparison and validation
- ✅ Feature importance assessment
- ✅ Avoiding false discoveries in ML

### **When NOT to Use**
- ❌ When temporal dependencies exist
- ❌ For precision-focused applications
- ❌ With extremely small sample sizes (< 30 per class)
- ❌ When class balance is artificially manipulated

## 🎯 **Summary**

This implementation provides a **theoretically sound, computationally efficient** way to establish recall baselines without running expensive simulations. The key insight that **expected recall equals signal proportion** under random prediction makes this approach both elegant and practical for ML model validation.
