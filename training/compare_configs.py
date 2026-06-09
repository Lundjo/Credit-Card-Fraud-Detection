from config import Config
from fraud_detection_system import FraudDetectionSystem


def get_configs():
    configs = {}

    # podrazumevana konfiguracija
    configs['default'] = Config()

    # full - vise stabala, veca dubina
    c = Config()
    c.SAMPLE_FRACTION = 1.0
    c.USE_BALANCING = True
    c.SMOTE_SAMPLING_STRATEGY = 0.4
    c.UNDERSAMPLING_STRATEGY = 0.6
    c.RF_N_ESTIMATORS = 200
    c.RF_MAX_DEPTH = 30
    c.ARF_N_MODELS = 20
    c.ARF_LAMBDA = 10
    configs['full'] = c

    # fast - manje stabala, plica
    c = Config()
    c.SAMPLE_FRACTION = 0.1
    c.USE_BALANCING = False
    c.RF_N_ESTIMATORS = 20
    c.RF_MAX_DEPTH = 10
    c.ARF_N_MODELS = 3
    c.ARF_LAMBDA = 3
    configs['fast'] = c

    # bez balansiranja - da vidimo uticaj SMOTE-a
    c = Config()
    c.USE_BALANCING = False
    configs['no_balancing'] = c

    # veci threshold - strozi u proglasavanju prevare (vise FN, manje FP)
    c = Config()
    c.ARF_THRESHOLD = 0.3
    configs['high_threshold'] = c

    # manji threshold - agresivniji (vise FP, manje FN)
    c = Config()
    c.ARF_THRESHOLD = 0.05
    configs['low_threshold'] = c

    # vise ARF modela
    c = Config()
    c.ARF_N_MODELS = 20
    configs['more_arf_models'] = c

    return configs


if __name__ == '__main__':
    configs = get_configs()
    total = len(configs)

    print(f"Running {total} configurations...\n")
    print("Configurations:")
    for name in configs:
        print(f"  - {name}")
    print()

    results_summary = []

    for i, (name, config) in enumerate(configs.items(), 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] Configuration: {name}")
        print(f"{'='*60}")

        try:
            system = FraudDetectionSystem(config, run_name=name)
            metrics = system.run_complete_pipeline(0, 2000)

            if metrics:
                results_summary.append({
                    'name': name,
                    'f1': metrics['f1'],
                    'recall': metrics['recall'],
                    'precision': metrics['precision'],
                    'auc': metrics['auc'],
                    'detected': metrics['detected_frauds'],
                    'total_fraud': metrics['fraud_count'],
                    'false_alarms': metrics['false_alarms'],
                })
                print(f"\nResult: F1={metrics['f1']:.4f} | "
                      f"Recall={metrics['recall']:.4f} | "
                      f"Precision={metrics['precision']:.4f} | "
                      f"AUC={metrics['auc']:.4f}")
        except Exception as e:
            print(f"Error in configuration '{name}': {e}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Configuration':<20} {'F1':>8} {'Recall':>8} {'Precision':>10} {'AUC':>8} {'Detected':>10}")
    print('-' * 70)
    for r in sorted(results_summary, key=lambda x: x['f1'], reverse=True):
        print(f"{r['name']:<20} {r['f1']:>8.4f} {r['recall']:>8.4f} "
              f"{r['precision']:>10.4f} {r['auc']:>8.4f} "
              f"{r['detected']:>4}/{r['total_fraud']:<5}")

    print(f"\nAll results saved to data/results.json")
