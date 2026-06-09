import shap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config
from data_loader import DataLoader
from initial_model import InitialModel

PLOTS_PATH = Path(__file__).parent.parent / 'data' / 'plots'
DATA_PATH = Path(__file__).parent.parent / 'data' / 'creditcard.csv'


def run_shap_analysis(config=None):
    if config is None:
        config = Config()

    PLOTS_PATH.mkdir(exist_ok=True)

    # ucitavanje i podela podataka
    print("Loading data...")
    loader = DataLoader(DATA_PATH, config.SAMPLE_FRACTION, config.RANDOM_SEED)
    loader.load_data()
    initial_data, _ = loader.split_data(config.INITIAL_SPLIT)

    feature_cols = [col for col in initial_data.columns if col not in ['Class', 'Time']]
    X = initial_data[feature_cols]
    y = initial_data['Class']

    # trening inicijalnog modela
    print("Training RF model...")
    model = InitialModel(config.USE_BALANCING, config)
    model.train(X, y)

    # shap koristi manji uzorak zbog brzine (ceo skup moze biti spor)
    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=config.RANDOM_SEED)

    print(f"Running SHAP analysis on {sample_size} samples...")
    explainer = shap.TreeExplainer(model.model)

    # novi shap API - vraca Explanation objekat sa pravilno definisanim dimenzijama
    explanation_obj = explainer(X_sample)

    # za binarnu klasifikaciju uzimamo fraud klasu (indeks 1)
    if explanation_obj.values.ndim == 3:
        fraud_explanation = explanation_obj[:, :, 1]
    else:
        fraud_explanation = explanation_obj

    # summary bar plot - prosecna apsolutna shap vrednost po featureu
    print("Saving summary bar plot...")
    plt.figure(figsize=(10, 7))
    shap.plots.bar(fraud_explanation, max_display=len(feature_cols), show=False)
    plt.title('Feature importance (mean |SHAP value|) — Fraud class')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'shap_summary_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_summary_bar.png")

    # beeswarm plot - distribucija shap vrednosti po featureu
    print("Saving beeswarm plot...")
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(fraud_explanation, max_display=len(feature_cols), show=False)
    plt.title('SHAP value distribution — Fraud class')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_beeswarm.png")

    # waterfall plot - uzimamo prvu fraud transakciju direktno iz initial_data
    fraud_rows = initial_data[initial_data['Class'] == 1]
    if len(fraud_rows) > 0:
        print("Saving waterfall plot for one fraud transaction...")

        x_fraud = fraud_rows[feature_cols].iloc[[0]]

        # racunamo shap vrednosti samo za tu jednu transakciju
        sv_single = explainer.shap_values(x_fraud)

        # podrska za razlicite formate u zavisnosti od verzije shap-a
        # stariji format: lista [klasa_0, klasa_1] gde je svaki (n_samples, n_features)
        # noviji format: niz oblika (n_samples, n_features, n_klasa)
        if isinstance(sv_single, list):
            values = sv_single[1][0]
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        elif sv_single.ndim == 3:
            values = sv_single[0, :, 1]
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            values = sv_single[0]
            base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

        explanation = shap.Explanation(
            values=values,
            base_values=float(base_value),
            data=x_fraud.values[0],
            feature_names=feature_cols
        )

        plt.figure(figsize=(10, 7))
        shap.waterfall_plot(explanation, show=False, max_display=15)
        plt.title('SHAP waterfall — single fraud transaction')
        plt.tight_layout()
        plt.savefig(PLOTS_PATH / 'shap_waterfall.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: shap_waterfall.png")
    else:
        print("No fraud transactions found in initial data, skipping waterfall plot")

    print(f"\nAll SHAP plots saved to: data/plots/")


if __name__ == '__main__':
    run_shap_analysis()
