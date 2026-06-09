import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.ticker as ticker

RESULTS_PATH = Path(__file__).parent.parent / 'data' / 'results.json'
PLOTS_PATH = Path(__file__).parent.parent / 'data' / 'plots'


def load_results():
    if not RESULTS_PATH.exists():
        print("No results.json found — run compare_configs.py or main.py first")
        return None
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_metrics_comparison(runs):
    metrics_to_plot = ['f1', 'recall', 'precision', 'auc']
    labels = {'f1': 'F1 Score', 'recall': 'Recall', 'precision': 'Precision', 'auc': 'AUC-ROC'}

    names = list(runs.keys())
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    for i, metric in enumerate(metrics_to_plot):
        values = [runs[n]['metrics'].get(metric, 0) for n in names]
        bars = ax.bar(x + i * width, values, width, label=labels[metric], color=colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Metric value')
    ax.set_title('Metrics comparison across configurations')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'metrics_comparison.png', dpi=150)
    plt.close()
    print("Saved: metrics_comparison.png")


def plot_fraud_detection(runs):
    names = list(runs.keys())
    detected = [runs[n]['metrics'].get('detected_frauds', 0) for n in names]
    missed = [runs[n]['metrics'].get('missed_frauds', 0) for n in names]
    false_alarms = [runs[n]['metrics'].get('false_alarms', 0) for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - width, detected, width, label='Detected frauds (TP)', color='#4CAF50', alpha=0.85)
    ax.bar(x, missed, width, label='Missed frauds (FN)', color='#F44336', alpha=0.85)
    ax.bar(x + width, false_alarms, width, label='False alarms (FP)', color='#FF9800', alpha=0.85)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Number of transactions')
    ax.set_title('Fraud detection results across configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'fraud_detection.png', dpi=150)
    plt.close()
    print("Saved: fraud_detection.png")


def plot_precision_recall_tradeoff(runs):
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    for (name, run), color in zip(runs.items(), colors):
        precision = run['metrics'].get('precision', 0)
        recall = run['metrics'].get('recall', 0)
        ax.scatter(recall, precision, s=100, color=color, zorder=5)
        ax.annotate(name, (recall, precision), textcoords='offset points',
                    xytext=(6, 4), fontsize=8, color=color)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall tradeoff across configurations')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.3, label='Precision = Recall')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'precision_recall_tradeoff.png', dpi=150)
    plt.close()
    print("Saved: precision_recall_tradeoff.png")


def plot_config_params(runs):
    params = ['ARF_N_MODELS', 'ARF_LAMBDA', 'ARF_THRESHOLD', 'RF_N_ESTIMATORS', 'RF_MAX_DEPTH']
    names = list(runs.keys())

    fig, axes = plt.subplots(1, len(params), figsize=(16, 5))
    fig.suptitle('Config parameters across configurations', fontsize=13)

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    for ax, param in zip(axes, params):
        values = [runs[n]['config'].get(param, 0) for n in names]
        bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85)
        ax.set_title(param, fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values or [1]),
                    f'{val}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'config_params.png', dpi=150)
    plt.close()
    print("Saved: config_params.png")


def plot_confusion_matrices(runs):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle('Confusion matrices', fontsize=13)

    for ax, (run_name, run) in zip(axes, runs.items()):
        m = run.get('metrics', {})
        tp = m.get('true_positives', 0)
        fp = m.get('false_positives', 0)
        fn = m.get('false_negatives', 0)
        tn = m.get('true_negatives', 0)

        matrix = np.array([[tn, fp], [fn, tp]])
        total = matrix.sum()
        im = ax.imshow(matrix, cmap='Blues')

        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                pct = val / total * 100 if total > 0 else 0
                ax.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                        fontsize=11, color='white' if matrix[i, j] > matrix.max() / 2 else 'black')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted\nLegit', 'Predicted\nFraud'])
        ax.set_yticklabels(['Actual\nLegit', 'Actual\nFraud'])
        ax.set_title(run_name, fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'confusion_matrices.png', dpi=150)
    plt.close()
    print("Saved: confusion_matrices.png")


def plot_batch_evolution(runs):
    # metrike koje pratimo kroz batcheve
    metrics = [
        ('f1', 'F1 Score', '#2196F3'),
        ('recall', 'Recall', '#4CAF50'),
        ('precision', 'Precision', '#FF9800'),
    ]

    # jedan grafik po metrici, sve konfiguracije na istom
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Model evolution through batches', fontsize=13)

    for ax, (metric, label, color) in zip(axes, metrics):
        plotted = False
        colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
        for (run_name, run), c in zip(runs.items(), colors):
            history = run.get('batch_history', [])
            if not history:
                continue
            batches = [b['batch'] for b in history]
            values = [b[metric] for b in history]
            ax.plot(batches, values, label=run_name, color=c, linewidth=1.2, alpha=0.85)
            plotted = True

        ax.set_ylabel(label)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=8, loc='lower right')

    axes[-1].set_xlabel('Batch')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'batch_evolution.png', dpi=150)
    plt.close()
    print("Saved: batch_evolution.png")


def plot_batch_frauds(runs):
    # broj detektovanih i propustenih prevara po batchu
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Fraud detection per batch', fontsize=13)

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for (run_name, run), c in zip(runs.items(), colors):
        history = run.get('batch_history', [])
        if not history:
            continue
        batches = [b['batch'] for b in history]
        detected = [b['detected_frauds'] for b in history]
        missed = [b['missed_frauds'] for b in history]

        axes[0].plot(batches, detected, label=run_name, color=c, linewidth=1.2)
        axes[1].plot(batches, missed, label=run_name, color=c, linewidth=1.2)

    axes[0].set_ylabel('Detected frauds (TP)')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_ylabel('Missed frauds (FN)')
    axes[1].set_xlabel('Batch')
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'batch_frauds.png', dpi=150)
    plt.close()
    print("Saved: batch_frauds.png")


def plot_radar(runs):
    metrics = ['f1', 'recall', 'precision', 'auc']
    labels = ['F1', 'Recall', 'Precision', 'AUC']
    n = len(metrics)

    # uglovi za svaku osu, zatvorimo krug ponavljanjem prvog
    angles = [i * 2 * np.pi / n for i in range(n)] + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for (run_name, run), color in zip(runs.items(), colors):
        m = run.get('metrics', {})
        values = [m.get(metric, 0) for metric in metrics] + [m.get(metrics[0], 0)]
        ax.plot(angles, values, color=color, linewidth=1.5, label=run_name)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title('Configuration comparison — radar chart', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: radar.png")


def plot_heatmap(runs):
    metrics = ['f1', 'recall', 'precision', 'auc', 'false_alarm_rate']
    labels = ['F1', 'Recall', 'Precision', 'AUC', 'False Alarm Rate']

    names = list(runs.keys())
    data = np.array([[runs[n]['metrics'].get(m, 0) for m in metrics] for n in names])

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.6 + 2)))

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)

    for i in range(len(names)):
        for j in range(len(metrics)):
            val = data[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color='black' if 0.25 < val < 0.75 else 'white')

    ax.set_title('Metrics heatmap across configurations', fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: heatmap.png")


if __name__ == '__main__':
    data = load_results()
    if data is None:
        exit(1)

    runs = data.get('runs', {})
    if not runs:
        print("No runs found in results.json")
        exit(1)

    PLOTS_PATH.mkdir(exist_ok=True)
    print(f"Generating plots for {len(runs)} configurations: {', '.join(runs.keys())}\n")

    plot_metrics_comparison(runs)
    plot_fraud_detection(runs)
    plot_precision_recall_tradeoff(runs)
    plot_confusion_matrices(runs)
    plot_radar(runs)
    plot_heatmap(runs)
    plot_batch_evolution(runs)
    plot_batch_frauds(runs)

    # config params samo ako ima vise runova
    if len(runs) > 1:
        configs_with_data = {n: r for n, r in runs.items() if r.get('config')}
        if configs_with_data:
            plot_config_params(configs_with_data)

    print(f"\nAll plots saved to: data/plots/")
