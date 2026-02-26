import json
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path


class MetricsTracker:
    def __init__(self,):
        self.metrics_history = []
        self.current_batch = 0

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

        # aproksimacija auc-a preko razlike prosecnih verovatnoca
        fraud_probs = [p for p, a in zip(probabilities, actuals) if a]
        legit_probs = [p for p, a in zip(probabilities, actuals) if not a]

        approx_auc = abs(np.mean(fraud_probs) - np.mean(legit_probs)) if (fraud_probs and legit_probs) else 0

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'fpr': float(fpr),
            'auc': float(approx_auc),
            'confusion_matrix': cm.tolist(),
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

    def save_to_file(self):
        data = {
            'export_time': datetime.now().isoformat(),
            'metrics_history': self.metrics_history
        }

        with open(Path(__file__).parent.parent / 'data' / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)