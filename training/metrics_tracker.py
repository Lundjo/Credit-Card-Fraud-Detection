import json
from datetime import datetime
from collections import deque
import numpy as np
from sklearn.metrics import confusion_matrix

class MetricsTracker:
    def __init__(self, fraud_buffer_size=100):
        self.metrics_history = []  # Lista svih metrika po batch-evima
        self.current_batch = 0  # Broj trenutnog batch-a

        # Buffer za prevare - čuva skorašnje prevare da bi model "pamtio"
        # retke prevare duže nego česte legitimne transakcije
        self.fraud_buffer = deque(maxlen=fraud_buffer_size)

    def add_initial_metrics(self, metrics, model_type='initial_rf'):
        metric_entry = {
            'timestamp': datetime.now().isoformat(),
            'batch': 0,  # Inicijalni model je batch 0
            'model_type': model_type,
            **metrics  # Dodaj sve metrike iz dict-a
        }

        self.metrics_history.append(metric_entry)
        print(f"\n✓ Inicijalne metrike sačuvane")

    def calculate_batch_metrics(self, predictions, actuals, probabilities, batch_num):
        # Konvertuj u numpy arrays
        y_true = np.array(actuals, dtype=int)
        y_pred = np.array(predictions, dtype=int)

        # Confusion matrix - uvek koristi obe labele
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Distribriraj vrednosti
        tn, fp, fn, tp = cm.ravel()

        # Kalkuliši metrike sa zaštitom od deljenja nulom
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total if total > 0 else 0

        # Precision - od svih predikovanih prevara, koliko je stvarno prevara?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall (Sensitivity) - od svih stvarnih prevara, koliko smo detektovali?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 Score - harmonijska sredina precision i recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # False Positive Rate - koliko legitimnih smo pogrešno označili kao prevare?
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # ROC-AUC - area under ROC curve
        # Za batch nivo koristimo prosečnu verovatnoću kao proxy
        fraud_probs = [p for p, a in zip(probabilities, actuals) if a]
        legit_probs = [p for p, a in zip(probabilities, actuals) if not a]

        # Proveri da li ima podataka za obe klase
        avg_fraud_proba = np.mean(fraud_probs) if fraud_probs else 0
        avg_legit_proba = np.mean(legit_probs) if legit_probs else 0

        # Approx AUC (nije egzaktan ali dobar indikator)
        if fraud_probs and legit_probs:
            approx_auc = abs(avg_fraud_proba - avg_legit_proba)
        else:
            approx_auc = 0

        # Kreiraj metric entry
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'batch': batch_num,
            'model_type': 'online_arf',

            # Osnovne metrike
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'fpr': float(fpr),
            'auc': float(approx_auc),

            # Confusion matrix
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),

            # Dodatne informacije
            'total_transactions': int(len(actuals)),
            'fraud_count': int(sum(actuals)),
            'detected_frauds': int(tp),
            'missed_frauds': int(fn),
            'false_alarms': int(fp),

            # Detection rate
            'detection_rate': float(recall),  # Isti kao recall
            'false_alarm_rate': float(fpr)
        }

        # Dodaj u istoriju
        self.metrics_history.append(metrics)

        return metrics

    def add_fraud_to_buffer(self, fraud_features):
        self.fraud_buffer.append({
            'timestamp': datetime.now().isoformat(),
            'features': fraud_features
        })

    def get_recent_frauds(self, n=10):
        return list(self.fraud_buffer)[-n:] if len(self.fraud_buffer) >= n else list(self.fraud_buffer)

    def get_metrics_summary(self):
        if not self.metrics_history:
            return None

        # Poslednje metrike (trenutno stanje)
        latest = self.metrics_history[-1]

        # Prosečne metrike preko svih batch-eva (osim inicijalnog)
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

        # Ukupne statistike
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

    def get_trend_analysis(self):
        online_metrics = [m for m in self.metrics_history if m['model_type'] == 'online_arf']

        if len(online_metrics) < 2:
            return {'status': 'insufficient_data'}

        # Uzmi prve 10% i poslednjih 10% batch-eva
        n = max(1, len(online_metrics) // 10)
        early_batches = online_metrics[:n]
        recent_batches = online_metrics[-n:]

        def avg_metric(batches, metric_name):
            values = [b[metric_name] for b in batches if metric_name in b]
            return np.mean(values) if values else 0

        # Poredi early vs recent
        trends = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            early_avg = avg_metric(early_batches, metric)
            recent_avg = avg_metric(recent_batches, metric)

            # Ako nema vrednosti, preskoči
            if early_avg == 0 and recent_avg == 0:
                continue

            change = recent_avg - early_avg
            change_pct = (change / early_avg * 100) if early_avg > 0 else 0

            if abs(change_pct) < 1:
                status = 'stable'
            elif change_pct > 0:
                status = 'improving'
            else:
                status = 'declining'

            trends[metric] = {
                'early_avg': float(early_avg),
                'recent_avg': float(recent_avg),
                'change': float(change),
                'change_pct': float(change_pct),
                'status': status
            }

        return trends

    def save_to_file(self, filepath='metrics_history.json'):
        data = {
            'export_time': datetime.now().isoformat(),
            'total_batches': len(self.metrics_history),
            'metrics_history': self.metrics_history,
            'summary': self.get_metrics_summary(),
            'trends': self.get_trend_analysis()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Metrike sačuvane u: {filepath}")

    def load_from_file(self, filepath='metrics_history.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.metrics_history = data['metrics_history']
        self.current_batch = len([m for m in self.metrics_history if m['model_type'] == 'online_arf'])

        print(f"✓ Metrike učitane iz: {filepath}")
        print(f"  Batch-eva: {self.current_batch}")