import json
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path


class MetricsTracker:
    def __init__(self,):
        self.metrics_history = []  # lista svih metrika po batchevima
        self.current_batch = 0  # broj trenutnog batcha

    # rezultati obicnog rf
    def add_initial_metrics(self, metrics, model_type):
        metric_entry = {
            'timestamp': datetime.now().isoformat(),  # kada je sacuvano
            'batch': 0,  # inicijalni model je batch 0
            'model_type': model_type,
            **metrics  # dodaje sve metrike u ovu novu
        }

        self.metrics_history.append(metric_entry)
        print(f"\n✓ Inicijalne metrike sačuvane")

    def calculate_batch_metrics(self, predictions, actuals, probabilities, batch_num):
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

        # roc kriva za odredjivanje kvaliteta modela
        fraud_probs = [p for p, a in zip(probabilities, actuals) if a]  # ako je prevara sacuvaj verovatnocu
        legit_probs = [p for p, a in zip(probabilities, actuals) if not a]  # verovatnoca za legitimnu

        # prosek treba biti sto veci za fraud, a sto manji za legit
        avg_fraud_proba = np.mean(fraud_probs) if fraud_probs else 0
        avg_legit_proba = np.mean(legit_probs) if legit_probs else 0

        if fraud_probs and legit_probs:
            approx_auc = abs(avg_fraud_proba - avg_legit_proba)
        else:
            approx_auc = 0

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'batch': batch_num,
            'model_type': 'online_arf',
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
            'total_transactions': int(len(actuals)),
            'fraud_count': int(sum(actuals)),
            'detected_frauds': int(tp),
            'missed_frauds': int(fn),
            'false_alarms': int(fp),
            'detection_rate': float(recall),
            'false_alarm_rate': float(fpr)
        }

        self.metrics_history.append(metrics)

        return metrics

    def get_metrics_summary(self):
        if not self.metrics_history:
            return None

        # poslednje metrike
        latest = self.metrics_history[-1]

        # prosecne metrike za online
        online_metrics = [m for m in self.metrics_history if m['model_type'] == 'online_arf']

        if online_metrics:
            avg_metrics = {
                'avg_accuracy': np.mean([m['accuracy'] for m in online_metrics]),
                'avg_precision': np.mean([m['precision'] for m in online_metrics]),
                'avg_recall': np.mean([m['recall'] for m in online_metrics]),
                'avg_f1': np.mean([m['f1'] for m in online_metrics]),
                'avg_fpr': np.mean([m['fpr'] for m in online_metrics])
            }
        else:
            avg_metrics = {}

        total_transactions = sum([m.get('total_transactions', 0) for m in online_metrics])
        total_frauds = sum([m.get('fraud_count', 0) for m in online_metrics])
        total_detected = sum([m.get('detected_frauds', 0) for m in online_metrics])
        total_missed = sum([m.get('missed_frauds', 0) for m in online_metrics])

        return {
            'current': latest,
            'averages': avg_metrics,
            'totals': {
                'transactions_processed': total_transactions,
                'frauds_encountered': total_frauds,
                'frauds_detected': total_detected,
                'frauds_missed': total_missed,
                'overall_detection_rate': total_detected / total_frauds if total_frauds > 0 else 0
            },
            'total_batches': len(online_metrics)
        }

    # da li je uopste potrebno ili bar ovoliko detaljno
    def get_trend_analysis(self):
        online_metrics = [m for m in self.metrics_history
                          if m['model_type'] == 'online_arf']

        if len(online_metrics) < 20:  # minimum za smislenu analizu
            return {'status': 'insufficient_data',
                    'message': f'Potrebno min 20 batch-eva, ima {len(online_metrics)}'}

        # rolling window
        window_size = max(5, len(online_metrics) // 10)

        trends = {}

        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = []
            for batch in online_metrics:
                if metric in batch:
                    values.append(batch[metric])

            if len(values) < window_size * 2:
                continue

            # rolling average na pocetku
            early_values = values[:window_size]
            early_avg = sum(early_values) / len(early_values)

            # rolling average na kraju
            recent_values = values[-window_size:]
            recent_avg = sum(recent_values) / len(recent_values)

            change = recent_avg - early_avg
            change_pct = (change / early_avg * 100) if early_avg > 0 else 0

            overall_avg = sum(values) / len(values)

            # standardna devijacija
            if len(values) > 1:
                variance = sum((x - overall_avg) ** 2 for x in values) / (len(values) - 1)
                std_dev = variance ** 0.5
            else:
                std_dev = 0

            if abs(change_pct) < 2:
                status = 'stable'
            elif change_pct > 5:
                status = 'improving_strongly'
            elif change_pct > 0:
                status = 'improving'
            elif change_pct < -5:
                status = 'declining_strongly'
            else:
                status = 'declining'

            trends[metric] = {
                'early_window_avg': early_avg,
                'recent_window_avg': recent_avg,
                'overall_avg': overall_avg,
                'change': change,
                'change_percent': change_pct,
                'volatility': std_dev / overall_avg * 100 if overall_avg > 0 else 0,
                'status': status,
                'samples': {
                    'total': len(values),
                    'window_size': window_size
                }
            }

        return trends

    def save_to_file(self):
        data = {
            'export_time': datetime.now().isoformat(),
            'total_batches': len(self.metrics_history),
            'metrics_history': self.metrics_history,
            'summary': self.get_metrics_summary(),
            'trends': self.get_trend_analysis()
        }

        with open(Path(__file__).parent.parent / 'data' / 'metrics_history.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
