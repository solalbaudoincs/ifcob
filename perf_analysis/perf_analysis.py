import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple


def recall(predictions: pd.Series, targets: pd.Series, signal: int) -> float:
    """
    Calculate recall performance metric.
    
    Args:
        predictions (pd.Series): Predicted values.
        targets (pd.Series): True target values.
        
    Returns:
        float: Recall performance metric.
    """
    true_positives = np.sum((predictions == signal) & (targets == signal))
    all_positives = np.sum(targets == signal)
    
    if all_positives == 0:
        return 0.0
    
    return true_positives / all_positives

def recall_interval_random_baseline(targets: pd.Series, signal: int, confidence: float = 0.95) -> tuple:
    """
    Calculate the confidence interval for recall under random baseline prediction.
    
    For a random predictor that follows the target distribution, the expected recall 
    for each signal equals the proportion of that signal in the targets.
    
    Args:
        targets (pd.Series): True target values.
        signal (int): The signal value (-1, 0, or 1) to calculate recall for.
        confidence (float): Confidence level for the interval.
        
    Returns:
        tuple: Lower and upper bounds of the confidence interval for recall.
    """
    
    total_samples = len(targets)
    actual_positives = np.sum(targets == signal)
    
    if actual_positives == 0:
        return 0.0, 0.0
    
    # Under random prediction matching target proportions:
    # P(predict signal) = proportion of signal in targets
    # Expected recall = P(predict signal | true signal) = P(predict signal) = proportion
    signal_proportion = actual_positives / total_samples
    
    # Expected true positives under random prediction
    expected_tp = actual_positives * signal_proportion
    
    # Use binomial confidence interval
    lower, upper = proportion_confint(
        count=expected_tp,
        nobs=actual_positives,
        alpha=1 - confidence,
        method='wilson'
    )
    
    return lower, upper

def compute_all_recall_intervals_random_baseline(targets: pd.Series, confidence: float = 0.95, verbose=False) -> dict:
    """
    Compute confidence intervals for recall of all signals under random baseline.
    
    Args:
        targets (pd.Series): True target values containing -1, 0, 1.
        confidence (float): Confidence level for the interval.
        
    Returns:
        dict: Dictionary containing recall intervals for each signal and summary statistics.
    """
    results = {}
    signal_counts = targets.value_counts()
    total_samples = len(targets)
    
    if verbose:
        print(f"🎯 RANDOM BASELINE RECALL CONFIDENCE INTERVALS")
        print(f"{'='*60}")
        print(f"Total samples: {total_samples:,}")
        print(f"Confidence level: {confidence*100:.1f}%")
        print(f"{'='*60}")
    
    # Calculate for each signal present in targets
    for signal in sorted(signal_counts.index):
        count = signal_counts[signal]
        proportion = count / total_samples
        
        # Expected recall under random prediction = signal proportion
        expected_recall = proportion
        
        # Confidence interval
        lower, upper = recall_interval_random_baseline(targets, signal, confidence)
        
        results[signal] = {
            'count': count,
            'proportion': proportion,
            'expected_recall': expected_recall,
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_width': upper - lower
        }
        if verbose:
            print(f"Signal {signal:2d}: count={count:6,} ({proportion:.3f}) | "
              f"Expected recall={expected_recall:.3f} | "
              f"CI=[{lower:.3f}, {upper:.3f}] | width={upper-lower:.3f}")
    if verbose:
        print(f"{'='*60}")
    
    # Add summary statistics
    results['summary'] = {
        'total_samples': total_samples,
        'unique_signals': len(signal_counts),
        'most_frequent_signal': signal_counts.idxmax(),
        'most_frequent_proportion': signal_counts.max() / total_samples,
        'least_frequent_signal': signal_counts.idxmin(),
        'least_frequent_proportion': signal_counts.min() / total_samples,
    }
    
    return results

def prediction_recall_significance(predictions: pd.Series, targets: pd.Series, confidence: float = 0.95) -> dict:
    """
    Check the predictions recall against the confidence intervals of the recalls
    under random baseline for all signals.
    
    Args:
        predictions (pd.Series): Predicted values.
        targets (pd.Series): True target values.
        confidence (float): Confidence level for the significance test.
        
    Returns:
        dict: Significance test results including recall and confidence interval.
    """
    recall_values = {signal: recall(predictions, targets, signal) for signal in [-1, 0, 1]}
    ci_results = compute_all_recall_intervals_random_baseline(targets, confidence)
    
    return {signal: {
        "significant": recall_values[signal] > ci_results[signal]["ci_upper"],  # Fixed the bug here
        'recall': recall_values[signal],
        'ci_lower': ci_results[signal]["ci_lower"],
        'ci_upper': ci_results[signal]["ci_upper"],
        'expected_random': ci_results[signal]["expected_recall"]
    } for signal in [-1, 0, 1]}

def plot_recall_confidence_intervals(targets: pd.Series, confidence: float = 0.95, 
                                   figsize: Tuple[int, int] = (12, 8), 
                                   title_prefix: str = "") -> plt.Figure:
    """
    Visualize recall confidence intervals for random baseline prediction.
    
    Args:
        targets (pd.Series): True target values containing -1, 0, 1.
        confidence (float): Confidence level for the interval.
        figsize (tuple): Figure size (width, height).
        title_prefix (str): Prefix for the plot title.
        
    Returns:
        plt.Figure: The matplotlib figure object.
    """
    results = compute_all_recall_intervals_random_baseline(targets, confidence)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{title_prefix}Random Baseline Recall Analysis (CI: {confidence*100:.1f}%)', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    signals = [-1, 0, 1]
    signal_names = ['Sell (-1)', 'Hold (0)', 'Buy (1)']
    colors = ['red', 'gray', 'green']
    
    expected_recalls = [results[signal]['expected_recall'] for signal in signals]
    ci_lowers = [results[signal]['ci_lower'] for signal in signals]
    ci_uppers = [results[signal]['ci_upper'] for signal in signals]
    counts = [results[signal]['count'] for signal in signals]
    proportions = [results[signal]['proportion'] for signal in signals]
    
    # Plot 1: Expected Recalls with Confidence Intervals
    ax1.bar(signal_names, expected_recalls, color=colors, alpha=0.7, edgecolor='black')
    ax1.errorbar(signal_names, expected_recalls, 
                yerr=[np.array(expected_recalls) - np.array(ci_lowers),
                      np.array(ci_uppers) - np.array(expected_recalls)],
                fmt='none', color='black', capsize=5, capthick=2)
    ax1.set_title('Expected Recall with Confidence Intervals', fontweight='bold')
    ax1.set_ylabel('Recall')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(ci_uppers) * 1.1)
    
    # Add value labels on bars
    for i, (recall, ci_lower, ci_upper) in enumerate(zip(expected_recalls, ci_lowers, ci_uppers)):
        ax1.text(i, recall + 0.01, f'{recall:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Signal Distribution
    ax2.pie(counts, labels=[f'{name}\n({count:,})' for name, count in zip(signal_names, counts)], 
            colors=colors, autopct='%1.2f%%', startangle=90)
    ax2.set_title('Signal Distribution', fontweight='bold')
    
    # Plot 3: Confidence Interval Widths
    ci_widths = [upper - lower for lower, upper in zip(ci_lowers, ci_uppers)]
    bars = ax3.bar(signal_names, ci_widths, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Confidence Interval Widths', fontweight='bold')
    ax3.set_ylabel('CI Width')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, width in zip(bars, ci_widths):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{width:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Summary Statistics
    ax4.axis('off')
    summary_text = f"""
    📊 SUMMARY STATISTICS
    {'='*30}
    Total Samples: {results['summary']['total_samples']:,}
    Unique Signals: {results['summary']['unique_signals']}
    
    Most Frequent Signal: {results['summary']['most_frequent_signal']} 
    ({results['summary']['most_frequent_proportion']:.3f})
    
    Least Frequent Signal: {results['summary']['least_frequent_signal']} 
    ({results['summary']['least_frequent_proportion']:.3f})
    
    Confidence Level: {confidence*100:.1f}%
    
    🎯 Under random baseline:
    Expected Recall = Signal Proportion
    """
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_prediction_performance(predictions: pd.Series, targets: pd.Series, 
                              confidence: float = 0.95, 
                              figsize: Tuple[int, int] = (15, 10),
                              title_prefix: str = "") -> plt.Figure:
    """
    Comprehensive visualization of prediction performance against random baseline.
    
    Args:
        predictions (pd.Series): Predicted values.
        targets (pd.Series): True target values.
        confidence (float): Confidence level for significance testing.
        figsize (tuple): Figure size (width, height).
        title_prefix (str): Prefix for the plot title.
        
    Returns:
        plt.Figure: The matplotlib figure object.
    """
    significance_results = prediction_recall_significance(predictions, targets, confidence)
    baseline_results = compute_all_recall_intervals_random_baseline(targets, confidence)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{title_prefix}Prediction Performance vs Random Baseline', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data
    signals = [-1, 0, 1]
    signal_names = ['Sell (-1)', 'Hold (0)', 'Buy (1)']
    colors = ['red', 'gray', 'green']
    
    actual_recalls = [significance_results[signal]['recall'] for signal in signals]
    expected_recalls = [significance_results[signal]['expected_random'] for signal in signals]
    ci_lowers = [significance_results[signal]['ci_lower'] for signal in signals]
    ci_uppers = [significance_results[signal]['ci_upper'] for signal in signals]
    is_significant = [significance_results[signal]['significant'] for signal in signals]
    
    # Plot 1: Actual vs Expected Recall Comparison
    x_pos = np.arange(len(signals))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, actual_recalls, width, label='Actual Recall', 
                    color=colors, alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, expected_recalls, width, label='Expected (Random)', 
                    color=colors, alpha=0.4, edgecolor='black', hatch='///')
    
    # Add confidence intervals for expected recalls
    ax1.errorbar(x_pos + width/2, expected_recalls, 
                yerr=[np.array(expected_recalls) - np.array(ci_lowers),
                      np.array(ci_uppers) - np.array(expected_recalls)],
                fmt='none', color='black', capsize=3, capthick=1)
    
    ax1.set_title('Actual vs Expected Recall', fontweight='bold')
    ax1.set_ylabel('Recall')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(signal_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add significance markers
    for i, (actual, significant) in enumerate(zip(actual_recalls, is_significant)):
        marker = '★' if significant else '○'
        color = 'gold' if significant else 'lightgray'
        ax1.text(i - width/2, actual + 0.02, marker, ha='center', va='bottom', 
                fontsize=16, color=color, fontweight='bold')
    
    # Add value labels
    for i, (actual, expected) in enumerate(zip(actual_recalls, expected_recalls)):
        ax1.text(i - width/2, actual + 0.01, f'{actual:.3f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
        ax1.text(i + width/2, expected + 0.01, f'{expected:.3f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    # Plot 2: Performance Improvement over Random
    improvements = [actual - expected for actual, expected in zip(actual_recalls, expected_recalls)]
    bar_colors = ['darkgreen' if imp > 0 else 'darkred' for imp in improvements]
    
    bars = ax2.bar(signal_names, improvements, color=bar_colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Recall Improvement over Random Baseline', fontweight='bold')
    ax2.set_ylabel('Recall Difference')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, imp, significant in zip(bars, improvements, is_significant):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        y_pos = height + (0.005 if height >= 0 else -0.005)
        significance_marker = ' ★' if significant else ''
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{imp:+.3f}{significance_marker}', ha='center', va=va, 
                fontweight='bold', fontsize=10)
    
    # Plot 3: Statistical Significance Summary
    ax3.axis('off')
    
    # Create significance summary
    total_signals = len(signals)
    significant_count = sum(is_significant)
    
    significance_text = f"""
    🎯 STATISTICAL SIGNIFICANCE ANALYSIS
    {'='*40}
    Confidence Level: {confidence*100:.1f}%
    
    Significant Signals: {significant_count}/{total_signals}
    
    """
    
    for signal, name, significant, actual, expected, ci_lower, ci_upper in zip(
        signals, signal_names, is_significant, actual_recalls, expected_recalls, ci_lowers, ci_uppers):
        
        status = "✅ SIGNIFICANT" if significant else "❌ Not Significant"
        significance_text += f"""
    {name}:
    Actual Recall: {actual:.4f}
    Expected: {expected:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]
    Status: {status}
    """
    
    ax3.text(0.05, 0.95, significance_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Plot 4: Confusion Matrix Style Visualization
    # Calculate prediction distribution
    pred_counts = predictions.value_counts().reindex(signals, fill_value=0)
    target_counts = targets.value_counts().reindex(signals, fill_value=0)
    
    # Create simple comparison
    comparison_data = pd.DataFrame({
        'Predictions': pred_counts.values,
        'Targets': target_counts.values
    }, index=signal_names)
    
    comparison_data.plot(kind='bar', ax=ax4, color=['lightblue', 'orange'], alpha=0.7)
    ax4.set_title('Prediction vs Target Distribution', fontweight='bold')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def theoretical_recall_distribution(targets: pd.Series, signal: int, n_simulations: int = 10000) -> np.ndarray:
    """
    Simulate the theoretical distribution of recall under random baseline prediction.
    
    This function validates our analytical confidence intervals by simulation.
    
    Args:
        targets (pd.Series): True target values.
        signal (int): The signal value to calculate recall distribution for.
        n_simulations (int): Number of simulations to run.
        
    Returns:
        np.ndarray: Array of recall values from simulations.
    """
    total_samples = len(targets)
    actual_positives = np.sum(targets == signal)
    signal_proportion = actual_positives / total_samples
    
    recall_values = []
    
    for _ in range(n_simulations):
        # Generate random predictions following target distribution
        random_predictions = np.random.choice(
            targets.unique(), 
            size=total_samples, 
            p=[np.sum(targets == val) / total_samples for val in targets.unique()]
        )
        
        # Calculate recall for this simulation
        true_positives = np.sum((random_predictions == signal) & (targets == signal))
        recall_sim = true_positives / actual_positives if actual_positives > 0 else 0
        recall_values.append(recall_sim)
    
    return np.array(recall_values)

def plot_theoretical_validation(targets: pd.Series, confidence: float = 0.95, 
                              n_simulations: int = 10000,
                              figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Validate analytical confidence intervals against theoretical simulation.
    
    Args:
        targets (pd.Series): True target values.
        confidence (float): Confidence level.
        n_simulations (int): Number of simulations for validation.
        figsize (tuple): Figure size.
        
    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Theoretical Validation: Analytical CI vs Simulation ({n_simulations:,} runs)', 
                 fontsize=14, fontweight='bold')
    
    signals = [-1, 0, 1]
    signal_names = ['Sell (-1)', 'Hold (0)', 'Buy (1)']
    colors = ['red', 'gray', 'green']
    
    baseline_results = compute_all_recall_intervals_random_baseline(targets, confidence)
    
    for i, (signal, name, color) in enumerate(zip(signals, signal_names, colors)):
        if signal not in targets.values:
            axes[i].text(0.5, 0.5, f'Signal {signal}\nnot present', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(name)
            continue
            
        # Get analytical results
        expected_recall = baseline_results[signal]['expected_recall']
        ci_lower = baseline_results[signal]['ci_lower']
        ci_upper = baseline_results[signal]['ci_upper']
        
        # Run simulation
        simulated_recalls = theoretical_recall_distribution(targets, signal, n_simulations)
        
        # Plot histogram of simulated recalls
        axes[i].hist(simulated_recalls, bins=50, alpha=0.7, color=color, 
                    density=True, edgecolor='black', linewidth=0.5)
        
        # Add analytical expected value and CI
        axes[i].axvline(expected_recall, color='red', linestyle='--', linewidth=2, 
                       label=f'Analytical Expected: {expected_recall:.3f}')
        axes[i].axvline(ci_lower, color='orange', linestyle=':', linewidth=2, 
                       label=f'Analytical CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        axes[i].axvline(ci_upper, color='orange', linestyle=':', linewidth=2)
        
        # Add simulation statistics
        sim_mean = np.mean(simulated_recalls)
        sim_ci_lower = np.percentile(simulated_recalls, (1-confidence)/2 * 100)
        sim_ci_upper = np.percentile(simulated_recalls, (1+confidence)/2 * 100)
        
        axes[i].axvline(sim_mean, color='blue', linestyle='-', linewidth=2, 
                       label=f'Simulation Mean: {sim_mean:.3f}')
        
        axes[i].set_title(name, fontweight='bold')
        axes[i].set_xlabel('Recall')
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
        
        # Add validation text
        validation_text = f"""
        Analytical: {expected_recall:.4f}
        Simulation: {sim_mean:.4f}
        Difference: {abs(expected_recall - sim_mean):.4f}
        """
        axes[i].text(0.02, 0.98, validation_text, transform=axes[i].transAxes, 
                    verticalalignment='top', fontsize=8, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

