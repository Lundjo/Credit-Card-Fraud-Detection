import time
from sklearn.model_selection import train_test_split
from config import Config
from data_loader import DataLoader
from initial_model import InitialModel
from online_model import OnlineModel
from metrics_tracker import MetricsTracker
from pathlib import Path
import pandas as pd

class FraudDetectionSystem:
    def __init__(self, data_path=Path(__file__).parent.parent / 'data' / 'creditcard.csv', config=None):
        self.config = config or Config()

        self.data_path = data_path
        self.initial_data = None
        self.streaming_data = None
        self.feature_names = None # obrisati?
        self.is_initialized = False
        self.is_trained = False

        self.data_loader = DataLoader(data_path=self.data_path, sample_fraction=self.config.SAMPLE_FRACTION, random_seed=self.config.RANDOM_SEED)
        self.initial_model = InitialModel(use_balancing=self.config.USE_BALANCING, config=self.config)
        self.online_model = OnlineModel(self.config, threshold=0.1)
        self.metrics_tracker = MetricsTracker()

        print("\n" + "=" * 70)
        print("  FRAUD DETECTION SYSTEM - INICIJALIZACIJA")
        print("=" * 70)
        print(f"📁 Dataset: {self.data_path}")
        print(f"📊 Sample: {self.config.SAMPLE_FRACTION * 100}% podataka")
        print(f"⚖️  Balansiranje: {'DA' if self.config.USE_BALANCING else 'NE'}")
        print(f"🌲 RF Stabla: {self.config.RF_N_ESTIMATORS}")
        print(f"🌲 ARF Stabla: {self.config.ARF_N_MODELS}")
        print("=" * 70)

    def load_and_prepare_data(self):
        print("\n" + "=" * 70)
        print("  KORAK 1: UČITAVANJE I PRIPREMA PODATAKA")
        print("=" * 70)

        self.data_loader.load_data()
        self.initial_data, self.streaming_data = self.data_loader.split_data(self.config.INITIAL_SPLIT)
        self.feature_names = self.data_loader.get_feature_names()

        self.is_initialized = True
        print("\n✓ Podaci uspešno učitani i podeljeni!")

        return self.initial_data, self.streaming_data

    def train_initial_model(self):
        if not self.is_initialized:
            raise ValueError("Sistem nije inicijalizovan! Pozovi load_and_prepare_data() prvo.")

        print("\n" + "=" * 70)
        print("  KORAK 2: TRENIRANJE INICIJALNOG MODELA")
        print("=" * 70)

        # pripremi podatke za trening
        X = self.initial_data.drop(['Class', 'Time'], axis=1)
        y = self.initial_data['Class']

        # train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=self.config.RANDOM_SEED,
            stratify=y  # odrzi proporciju prevara u oba skupa
        )

        print(f"\nTrain set: {len(X_train):,} transakcija")
        print(f"Validation set: {len(X_val):,} transakcija")

        # treniraj model
        results = self.initial_model.train(X_train, y_train, X_val, y_val)

        # sacuvaj metrike inicijalnog modela
        self.metrics_tracker.add_initial_metrics(results, model_type='initial_rf')

        # sacuvaj model
        self.initial_model.save(Path(__file__).parent.parent / 'data' / 'initial_rf_model.pkl')

        self.is_trained = True
        print("\n✓ Inicijalni model uspešno istreniran!")

        return results

    def create_warmup_data(self, initial_data, warmup_samples):
        legit = initial_data[initial_data['Class'] == 0]
        fraud = initial_data[initial_data['Class'] == 1]

        rows = len(initial_data)

        # ako ima malo legitimnih uzmi sve ili polovinu od ukupno, u suprotnom proporcionalan udeo da ostane
        if len(legit) < warmup_samples:
            legit_num = min(len(legit), warmup_samples // 2)
        else:
            legit_num = warmup_samples * len(legit) // rows

        # isti nacin uzimanja kao za legitimne
        if len(fraud) < warmup_samples:
            fraud_num = min(len(fraud), warmup_samples // 2)
        else:
            fraud_num = warmup_samples * len(fraud) // rows

        # uzmi random uzorke i spoji u jedinstven dataframe
        legit_sample = legit.sample(n=legit_num, random_state=self.config.RANDOM_SEED)
        fraud_sample = fraud.sample(n=fraud_num, random_state=self.config.RANDOM_SEED)
        warmup_data = pd.concat([legit_sample, fraud_sample], ignore_index=True)

        return warmup_data

    def initialize_online_model(self, warmup_samples=2000):
        if not self.is_trained:
            print("⚠️  Inicijalni model nije treniran, ali nastavljamo sa online modelom...")

        print("\n" + "=" * 70)
        print("  KORAK 3: INICIJALIZACIJA ONLINE MODELA (Warm-Start)")
        print("=" * 70)

        # inicijalizuj prazan ARF model
        self.online_model.initialize()

        # online model prvo uci od obicnog
        if warmup_samples > 0 and self.initial_data is not None:
            print(f"\n{'=' * 70}")
            print(f"🔥 WARM-UP FAZA: ARF uči od RF modela")
            print(f"{'=' * 70}")
            print(f"Broj primera za warm-up: {warmup_samples}")

            # pravljenje warmup dataseta
            warmup_data = self.create_warmup_data(self.initial_data, warmup_samples)

            feature_cols = [col for col in warmup_data.columns if col not in ['Class', 'Time']]
            fraud_count = 0
            legit_count = 0

            # predikcija obicnog modela
            X_warmup = warmup_data[feature_cols]
            rf_labels = self.initial_model.predict(X_warmup)

            print(f"\nUčenje u toku...")
            for i, (idx, row) in enumerate(warmup_data.iterrows()):
                x_dict = {col: float(row[col]) for col in feature_cols} # pretvara se u dict koji je ocekivani format
                y_true = bool(row['Class'])
                rf_label = bool(rf_labels[i]) # sta je obican model pretpostavio

                # poziv samo ugradjene metode jer se iskljucivo uci, ne evaluira se
                self.online_model.model.learn_one(x_dict, rf_label)

                if y_true:
                    fraud_count += 1
                else:
                    legit_count += 1

            print(f"\n\n✓ Warm-up uspešno završen!")
            print(f"  Ukupno naučeno: {len(warmup_data):,} primera")
            print(f"  - Legitimne transakcije: {legit_count:,}")
            print(f"  - Prevare: {fraud_count:,}")
            print(f"\n💡 ARF model sada ima bazno znanje iz RF modela!")
        else:
            print("\n⚠️  Warm-up preskoćen (warmup_samples=0)")

        print("\n✓ Online model spreman za streaming!")
        return self.online_model

    def simulate_streaming(self, delay=0):
        if not self.is_initialized:
            raise ValueError("Sistem nije inicijalizovan!")

        if self.online_model.model is None:
            raise ValueError("Online model nije inicijalizovan!")

        batch_size = self.config.DEFAULT_BATCH_SIZE

        print("\n" + "=" * 70)
        print("  KORAK 4: STREAMING SIMULACIJA")
        print("=" * 70)
        print(f"📦 Batch veličina: {batch_size}")
        print(f"🔄 Ukupno batch-eva: {len(self.streaming_data) // batch_size}")
        print(f"⏱️  Delay: {delay}s")
        print("=" * 70 + "\n")

        feature_cols = [col for col in self.streaming_data.columns
                        if col not in ['Class', 'Time']]

        batch_results = []
        batch_num = 0

        # batch grupisanje umesto na jednom primeru
        for i in range(0, len(self.streaming_data), batch_size):
            batch_num += 1
            batch_data = self.streaming_data.iloc[i:i + batch_size]

            batch_predictions = []
            batch_actuals = []
            batch_probabilities = []

            # predvidjanje na pojedinacnom primeru iz batcha
            for idx, row in batch_data.iterrows():
                x_dict = {col: float(row[col]) for col in feature_cols}
                y_true = bool(row['Class'])

                # prvo predikcija
                y_pred = self.online_model.predict_one(x_dict)
                y_pred_proba_dict = self.online_model.predict_proba_one(x_dict)
                y_pred_proba = y_pred_proba_dict.get(True, 0)

                # cuva se za batch metrike
                batch_predictions.append(y_pred)
                batch_actuals.append(y_true)
                batch_probabilities.append(y_pred_proba)

                # model uci na tek prediktovanom
                self.online_model.learn_one(x_dict, y_true)

            batch_metrics = self.metrics_tracker.calculate_batch_metrics(
                predictions=batch_predictions,
                actuals=batch_actuals,
                probabilities=batch_probabilities,
                batch_num=batch_num
            )

            batch_results.append(batch_metrics)

            print(f"Batch {batch_num:3d} | "
                  f"Acc: {batch_metrics['accuracy']:.4f} | "
                  f"Prec: {batch_metrics['precision']:.4f} | "
                  f"Rec: {batch_metrics['recall']:.4f} | "
                  f"F1: {batch_metrics['f1']:.4f} | "
                  f"Frauds: {batch_metrics['fraud_count']:2d}/{batch_metrics['total_transactions']:4d} | "
                  f"Detected: {batch_metrics['detected_frauds']:2d}")

            # delay za real time simulaciju
            if delay > 0:
                time.sleep(delay)

        print("\n" + "=" * 70)
        print("  STREAMING ZAVRŠEN")
        print("=" * 70)

        summary = self.metrics_tracker.get_metrics_summary()
        if summary:
            print(f"\n📊 UKUPNO PROCESOVANO:")
            print(f"  Transakcija: {summary['totals']['transactions_processed']:,}")
            print(f"  Prevara: {summary['totals']['frauds_encountered']:,}")
            print(f"  Detektovano: {summary['totals']['frauds_detected']:,}")
            print(f"  Propušteno: {summary['totals']['frauds_missed']:,}")
            print(f"  Detection Rate: {summary['totals']['overall_detection_rate'] * 100:.2f}%")

            print(f"\n📈 PROSEČNE METRIKE:")
            if summary['averages']:
                print(f"  Accuracy: {summary['averages']['avg_accuracy'] * 100:.2f}%")
                print(f"  Precision: {summary['averages']['avg_precision'] * 100:.2f}%")
                print(f"  Recall: {summary['averages']['avg_recall'] * 100:.2f}%")
                print(f"  F1-Score: {summary['averages']['avg_f1'] * 100:.2f}%")

        return batch_results

    def get_current_status(self):
        return {
            'initialized': self.is_initialized,
            'trained': self.is_trained,
            'current_batch': self.metrics_tracker.current_batch,
            'config': {
                'sample_fraction': self.config.SAMPLE_FRACTION,
                'use_balancing': self.config.USE_BALANCING,
                'batch_size': self.config.DEFAULT_BATCH_SIZE
            }
        }

    def save_all(self, metrics_file='metrics_history.json'):
        print("\n" + "=" * 70)
        print("  ČUVANJE SISTEMA")
        print("=" * 70)

        self.metrics_tracker.save_to_file(metrics_file)

        print("\n✓ Svi podaci sačuvani!")

    def run_complete_pipeline(self, streaming_delay=0, warmup_samples=2000):
        print("\n" + "🚀" * 35)
        print("  POKRETANJE KOMPLETNOG FRAUD DETECTION PIPELINE-A")
        print("  (RF → ARF Warm-Start → Streaming)")
        print("🚀" * 35 + "\n")

        start_time = time.time()

        try:
            self.load_and_prepare_data()
            initial_results = self.train_initial_model()
            self.initialize_online_model(warmup_samples=warmup_samples)
            streaming_results = self.simulate_streaming(delay=streaming_delay)

            self.save_all(Path(__file__).parent.parent / 'data' / 'metrics_history.json')

            end_time = time.time()
            elapsed = end_time - start_time

            report = {
                'success': True,
                'elapsed_time_seconds': elapsed,
                'configuration': {
                    'warmup_samples': warmup_samples,
                    'batch_size': self.config.DEFAULT_BATCH_SIZE,
                    'use_balancing': self.config.USE_BALANCING
                },
                'initial_model_results': initial_results,
                'streaming_summary': self.metrics_tracker.get_metrics_summary(),
                'trend_analysis': self.metrics_tracker.get_trend_analysis(),
                'system_status': self.get_current_status()
            }

            print(f"\n⏱️  Ukupno vreme: {elapsed:.2f} sekundi")
            print("✅ PIPELINE USPEŠNO ZAVRŠEN!\n")

            return report

        except Exception as e:
            print(f"\n❌ GREŠKA: {str(e)}")
            raise