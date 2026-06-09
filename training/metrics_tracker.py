import json
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from pathlib import Path


class MetricsTracker:
    def __init__(self):
        self.metrics_history = []
        self.batch_history = []
        self.current_batch = 0
        self.results_path = Path(__file__).parent.parent / 'data' / 'results.json'

    def track_batch(self, batch_num, predictions, actuals):
        tp = sum(1 for p, a in zip(predictions, actuals) if p and a)
        fp = sum(1 for p, a in zip(predictions, actuals) if p and not a)
        fn = sum(1 for p, a in zip(predictions, actuals) if not p and a)
        tn = sum(1 for p, a in zip(predictions, actuals) if not p and not a)
        total = len(actuals)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0

        self.batch_history.append({
            'batch': batch_num,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'detected_frauds': int(tp),
            'missed_frauds': int(fn),
            'false_alarms': int(fp),
            'fraud_count': int(sum(actuals)),
            'total': int(total),
        })

    def calculate_final_metrics(self, predictions, actuals, probabilities):
        # pretvara se u niz zbog brzine
        y_true = np.array(actuals, dtype=int)
        y_pred = np.array(predictions, dtype=int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        total = tn + fp + fn + tp

        # koliko ima tacnih predvidjanja
        accuracy = (tn + tp) / total if total > 0 else 0

        # od svih predikovanih prevara koliko je stvarno prevara
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # od svih stvarnih prevara koliko je detektovano
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # harmonijska sredina precision i recall koja detektuje da li je neki od njih ekstremno losiji
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # false positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # vrednost ispod roc krive, kolika je verovatnoca da ce dodeliti pravoj prevari tacnost
        auc = roc_auc_score(y_true, probabilities) if len(set(actuals)) > 1 else 0

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'fpr': float(fpr),
            'auc': float(auc),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_transactions': int(total),
            'fraud_count': int(sum(actuals)),
            'detected_frauds': int(tp),
            'missed_frauds': int(fn),
            'false_alarms': int(fp),
            'detection_rate': float(recall),
            'false_alarm_rate': float(fpr)
        }

        self.metrics_history.append(metrics)

        return metrics

    def save_to_file(self, run_name='default', config=None):
        # ucitaj postojece rezultate ako postoje
        if self.results_path.exists():
            with open(self.results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # podrzi stari format koji nema 'runs' kljuc
            all_results = data if 'runs' in data else {'runs': {}}
        else:
            all_results = {'runs': {}}

        # izvuci config parametre ako je prosledjen
        config_params = {}
        if config is not None:
            excluded = {'RF_N_JOBS', 'RANDOM_SEED'}
            for attr in dir(config):
                if attr in excluded or attr.startswith('_'):
                    continue
                value = getattr(config, attr)
                if type(value) in (int, float, bool, str):
                    config_params[attr] = value

        # sacuvaj run pod zadatim imenom (prepisuje ako vec postoji)
        all_results['runs'][run_name] = {
            'timestamp': datetime.now().isoformat(),
            'config': config_params,
            'metrics': self.metrics_history[-1] if self.metrics_history else {},
            'batch_history': self.batch_history
        }

        with open(self.results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)