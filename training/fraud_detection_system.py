import time
from sklearn.model_selection import train_test_split
from config import Config
from data_loader import DataLoader
from initial_model import InitialModel
from online_model import OnlineModel
from metrics_tracker import MetricsTracker
from pathlib import Path


class FraudDetectionSystem:
    def __init__(self,
                 data_path=Path(__file__).parent.parent / 'data' / 'creditcard.csv',
                 sample_fraction=1.0,
                 use_balancing=True,
                 config=None):
        """
        Inicijalizuje sistem sa željenim parametrima.

        Args:
            data_path (str): Putanja do CSV fajla
            sample_fraction (float): Koliko podataka koristiti (0.1-1.0)
            use_balancing (bool): Da li balansirati podatke sa SMOTE?
            config (Config): Custom konfiguracija (ili None za default)
        """
        # Konfiguracioni objekat
        self.config = config or Config()

        # Override config parametara ako su prosleđeni
        self.config.DATA_PATH = data_path
        self.config.SAMPLE_FRACTION = sample_fraction
        self.config.USE_BALANCING = use_balancing

        # Komponente sistema
        self.data_loader = DataLoader(
            data_path=self.config.DATA_PATH,
            sample_fraction=self.config.SAMPLE_FRACTION,
            random_seed=self.config.RANDOM_SEED
        )

        self.initial_model = InitialModel(
            use_balancing=self.config.USE_BALANCING,
            config=self.config
        )

        self.online_model = OnlineModel(config=self.config)

        self.metrics_tracker = MetricsTracker(
            fraud_buffer_size=self.config.FRAUD_BUFFER_SIZE
        )

        # Podaci
        self.initial_data = None
        self.streaming_data = None
        self.feature_names = None

        # Status
        self.is_initialized = False
        self.is_trained = False

        print("\n" + "=" * 70)
        print("  FRAUD DETECTION SYSTEM - INICIJALIZACIJA")
        print("=" * 70)
        print(f"📁 Dataset: {self.config.DATA_PATH}")
        print(f"📊 Sample: {self.config.SAMPLE_FRACTION * 100}% podataka")
        print(f"⚖️  Balansiranje: {'DA' if self.config.USE_BALANCING else 'NE'}")
        print(f"🌲 RF Stabla: {self.config.RF_N_ESTIMATORS}")
        print(f"🌲 ARF Stabla: {self.config.ARF_N_MODELS}")
        print("=" * 70)

    def load_and_prepare_data(self):
        """
        Učitava podatke i priprema ih za trening i streaming.

        Ova metoda:
        1. Učitava CSV fajl
        2. Primenjuje sampling ako je potrebno
        3. Deli podatke na inicijalni (70%) i streaming (30%) set

        Returns:
            tuple: (initial_data, streaming_data)
        """
        print("\n" + "=" * 70)
        print("  KORAK 1: UČITAVANJE I PRIPREMA PODATAKA")
        print("=" * 70)

        # Učitaj podatke
        self.data_loader.load_data()

        # Podeli na initial i streaming
        self.initial_data, self.streaming_data = self.data_loader.split_data(
            initial_split=self.config.INITIAL_SPLIT
        )

        # Sačuvaj nazive feature-a
        self.feature_names = self.data_loader.get_feature_names()

        self.is_initialized = True
        print("\n✓ Podaci uspešno učitani i podeljeni!")

        return self.initial_data, self.streaming_data

    def train_initial_model(self):
        """
        Trenira inicijalni Random Forest model.

        Ova metoda:
        1. Deli inicijalne podatke na train/validation
        2. (Opciono) Balansira podatke sa SMOTE
        3. Trenira Random Forest
        4. Evaluira na validation setu
        5. Sačuvava metrike

        Returns:
            dict: Rezultati evaluacije
        """
        if not self.is_initialized:
            raise ValueError("Sistem nije inicijalizovan! Pozovi load_and_prepare_data() prvo.")

        print("\n" + "=" * 70)
        print("  KORAK 2: TRENIRANJE INICIJALNOG MODELA")
        print("=" * 70)

        # Pripremi podatke za trening
        X = self.initial_data.drop(['Class', 'Time'], axis=1)
        y = self.initial_data['Class']

        # Train/Validation split (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=self.config.RANDOM_SEED,
            stratify=y  # Održi proporciju prevara
        )

        print(f"\nTrain set: {len(X_train):,} transakcija")
        print(f"Validation set: {len(X_val):,} transakcija")

        # Treniraj model
        results = self.initial_model.train(X_train, y_train, X_val, y_val)

        # Sačuvaj metrike inicijalnog modela
        self.metrics_tracker.add_initial_metrics(results, model_type='initial_rf')

        # Sačuvaj model na disk
        self.initial_model.save(Path(__file__).parent.parent / 'data' / 'initial_rf_model.pkl')

        self.is_trained = True
        print("\n✓ Inicijalni model uspešno istreniran!")

        return results

    def initialize_online_model(self, warmup_samples=2000):
        """
        Inicijalizuje online learning model (Adaptive Random Forest).

        WARM-START PRISTUP:
        ARF počinje učenje na inicijalnim podacima pre streaming-a.
        Ovo omogućava modelu da brže nauči distribuciju i smanji
        cold-start problem.

        Args:
            warmup_samples (int): Broj primera za warm-up (0 = bez warm-up)

        Returns:
            OnlineModel: Inicijalizovani online model
        """
        if not self.is_trained:
            print("⚠️  Inicijalni model nije treniran, ali nastavljamo sa online modelom...")

        print("\n" + "=" * 70)
        print("  KORAK 3: INICIJALIZACIJA ONLINE MODELA (Warm-Start)")
        print("=" * 70)

        # Inicijalizuj prazan ARF model
        self.online_model.initialize()

        # 🔥 WARM-UP FAZA: ARF uči na inicijalnim podacima
        if warmup_samples > 0 and self.initial_data is not None:
            print(f"\n{'=' * 70}")
            print(f"🔥 WARM-UP FAZA: ARF uči od RF modela")
            print(f"{'=' * 70}")
            print(f"Broj primera za warm-up: {warmup_samples}")

            # Uzmi stratified sample iz inicijalnih podataka
            # Ovo osigurava da imamo proporciju prevara kao u originalnom setu
            warmup_data = self.initial_data.groupby('Class', group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), warmup_samples // 2) if len(x) < warmup_samples else warmup_samples * len(x) // len(
                        self.initial_data),
                    random_state=self.config.RANDOM_SEED
                )
            ).reset_index(drop=True)

            # Pripremi feature-e
            feature_cols = [col for col in warmup_data.columns
                            if col not in ['Class', 'Time']]

            fraud_count = 0
            legit_count = 0

            print(f"\nUčenje u toku...")
            for idx, row in warmup_data.iterrows():
                x_dict = {col: float(row[col]) for col in feature_cols}
                y_true = bool(row['Class'])

                # ARF uči na ovom primeru (bez evaluacije - samo učenje)
                self.online_model.model.learn_one(x_dict, y_true)

                if y_true:
                    fraud_count += 1
                else:
                    legit_count += 1

                # Progress bar
                if (idx + 1) % 500 == 0:
                    print(f"  Procesovano: {idx + 1}/{len(warmup_data)} primera", end='\r')

            print(f"\n\n✓ Warm-up uspešno završen!")
            print(f"  Ukupno naučeno: {len(warmup_data):,} primera")
            print(f"  - Legitimne transakcije: {legit_count:,}")
            print(f"  - Prevare: {fraud_count:,}")
            print(f"\n💡 ARF model sada ima bazno znanje iz RF modela!")
        else:
            print("\n⚠️  Warm-up preskoćen (warmup_samples=0)")

        print("\n✓ Online model spreman za streaming!")
        return self.online_model

    def simulate_streaming(self, batch_size=None, delay=0, verbose=True):
        """
        Simulira streaming transakcija i online učenje.

        Ova metoda:
        1. Procesira streaming podatke batch-by-batch
        2. Za svaku transakciju:
           - Predvidi sa ARF modelom (koji je već naučio od RF-a)
           - Evaluira predikciju
           - Uči na toj transakciji (inkremantalno)
        3. Prati metrike po batch-evima
        4. Čuva prevare u buffer

        Args:
            batch_size (int): Veličina batch-a (None = koristi config default)
            delay (float): Pauza između batch-eva u sekundama (za vizuelizaciju)
            verbose (bool): Da li prikazivati progress

        Returns:
            list: Lista metrika po batch-evima
        """
        if not self.is_initialized:
            raise ValueError("Sistem nije inicijalizovan!")

        if self.online_model.model is None:
            raise ValueError("Online model nije inicijalizovan!")

        batch_size = batch_size or self.config.DEFAULT_BATCH_SIZE

        print("\n" + "=" * 70)
        print("  KORAK 4: STREAMING SIMULACIJA")
        print("=" * 70)
        print(f"📦 Batch veličina: {batch_size}")
        print(f"🔄 Ukupno batch-eva: {len(self.streaming_data) // batch_size}")
        print(f"⏱️  Delay: {delay}s")
        print("=" * 70 + "\n")

        # Pripremi feature nazive
        feature_cols = [col for col in self.streaming_data.columns
                        if col not in ['Class', 'Time']]

        batch_results = []
        batch_num = 0

        # Procesuj podatke batch-by-batch
        for i in range(0, len(self.streaming_data), batch_size):
            batch_num += 1
            batch_data = self.streaming_data.iloc[i:i + batch_size]

            # Liste za čuvanje rezultata batch-a
            batch_predictions = []
            batch_actuals = []
            batch_probabilities = []

            # === PROCESIRANJE SVAKE TRANSAKCIJE U BATCH-U ===
            for idx, row in batch_data.iterrows():
                # Konvertuj red u dictionary (format koji river zahteva)
                x_dict = {col: float(row[col]) for col in feature_cols}
                y_true = bool(row['Class'])

                # 1. PREDIKCIJA sa ARF modelom (koji je već warm-up-ovan)
                y_pred = self.online_model.predict_one(x_dict)
                y_pred_proba_dict = self.online_model.predict_proba_one(x_dict)
                y_pred_proba = y_pred_proba_dict.get(True, 0)

                # Sačuvaj za batch metrike
                batch_predictions.append(y_pred)
                batch_actuals.append(y_true)
                batch_probabilities.append(y_pred_proba)

                # 2. ONLINE UČENJE
                # Model nastavlja da uči na ovoj transakciji (incrementalno)
                self.online_model.learn_one(x_dict, y_true)

                # 3. AKO JE PREVARA - dodaj u buffer
                if y_true:
                    self.metrics_tracker.add_fraud_to_buffer(x_dict)

            # === KALKULACIJA METRIKA ZA BATCH ===
            batch_metrics = self.metrics_tracker.calculate_batch_metrics(
                predictions=batch_predictions,
                actuals=batch_actuals,
                probabilities=batch_probabilities,
                batch_num=batch_num
            )

            batch_results.append(batch_metrics)

            # === PROGRESS REPORT ===
            if verbose:
                print(f"Batch {batch_num:3d} | "
                      f"Acc: {batch_metrics['accuracy']:.4f} | "
                      f"Prec: {batch_metrics['precision']:.4f} | "
                      f"Rec: {batch_metrics['recall']:.4f} | "
                      f"F1: {batch_metrics['f1']:.4f} | "
                      f"Frauds: {batch_metrics['fraud_count']:2d}/{batch_metrics['total_transactions']:4d} | "
                      f"Detected: {batch_metrics['detected_frauds']:2d}")

            # Delay između batch-eva (za real-time simulaciju)
            if delay > 0:
                time.sleep(delay)

        print("\n" + "=" * 70)
        print("  STREAMING ZAVRŠEN")
        print("=" * 70)

        # Prikazi finalni summary
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
        """
        Vraća trenutni status sistema.

        Returns:
            dict: Status sa svim relevantnim informacijama
        """
        return {
            'initialized': self.is_initialized,
            'trained': self.is_trained,
            'current_batch': self.metrics_tracker.current_batch,
            'total_frauds_in_buffer': len(self.metrics_tracker.fraud_buffer),
            'config': {
                'sample_fraction': self.config.SAMPLE_FRACTION,
                'use_balancing': self.config.USE_BALANCING,
                'batch_size': self.config.DEFAULT_BATCH_SIZE
            }
        }

    def save_all(self, metrics_file='metrics_history.json'):
        """
        Sačuvaj sve komponente sistema.

        Args:
            metrics_file (str): Ime fajla za metrike
        """
        print("\n" + "=" * 70)
        print("  ČUVANJE SISTEMA")
        print("=" * 70)

        # Sačuvaj model (već sačuvan tokom treninga)
        # self.initial_model.save()

        # Sačuvaj metrike
        self.metrics_tracker.save_to_file(metrics_file)

        print("\n✓ Svi podaci sačuvani!")

    def run_complete_pipeline(self,
                              batch_size=None,
                              streaming_delay=0,
                              save_results=True,
                              warmup_samples=2000):
        """
        Pokreće kompletan workflow od početka do kraja.

        Ova metoda izvršava sve korake:
        1. Učitavanje podataka
        2. Trening inicijalnog RF modela
        3. Warm-start ARF modela sa znanjem iz RF-a
        4. Streaming simulacija (ARF nastavlja da uči)
        5. (Opciono) Čuvanje rezultata

        Args:
            batch_size (int): Veličina batch-a za streaming
            streaming_delay (float): Pauza između batch-eva
            save_results (bool): Da li sačuvati rezultate
            warmup_samples (int): Broj primera za ARF warm-up (preporuka: 2000-5000)

        Returns:
            dict: Kompletan izveštaj sa svim rezultatima
        """
        print("\n" + "🚀" * 35)
        print("  POKRETANJE KOMPLETNOG FRAUD DETECTION PIPELINE-A")
        print("  (RF → ARF Warm-Start → Streaming)")
        print("🚀" * 35 + "\n")

        start_time = time.time()

        try:
            # KORAK 1: Učitaj i pripremi podatke
            self.load_and_prepare_data()

            # KORAK 2: Treniraj inicijalni RF model
            initial_results = self.train_initial_model()

            # KORAK 3: Warm-start ARF sa znanjem iz RF
            self.initialize_online_model(warmup_samples=warmup_samples)

            # KORAK 4: Simuliraj streaming (ARF nastavlja da uči)
            streaming_results = self.simulate_streaming(
                batch_size=batch_size,
                delay=streaming_delay,
                verbose=True
            )

            # KORAK 5: Sačuvaj rezultate
            if save_results:
                self.save_all(Path(__file__).parent.parent / 'data' / 'metrics_history.json')

            end_time = time.time()
            elapsed = end_time - start_time

            # Kreiraj kompletan report
            report = {
                'success': True,
                'elapsed_time_seconds': elapsed,
                'configuration': {
                    'warmup_samples': warmup_samples,
                    'batch_size': batch_size or self.config.DEFAULT_BATCH_SIZE,
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


# ============================================================================
# STANDALONE EXECUTION - Za testiranje bez API-ja
# ============================================================================

if __name__ == '__main__':
    """
    Ovaj blok se izvršava kada direktno pokreneš ovaj fajl.
    Korisno za testiranje bez Flask API-ja.

    NAPOMENA: Sada ARF koristi warm-start - uči od RF modela pre streaming-a!
    """

    print("\n" + "=" * 70)
    print("  STANDALONE REŽIM - Testiranje sistema sa Warm-Start")
    print("=" * 70 + "\n")

    # Kreiraj custom config ako želiš (ili koristi default)
    custom_config = Config()
    custom_config.SAMPLE_FRACTION = 0.2  # Koristi samo 20% podataka za brže testiranje
    custom_config.USE_BALANCING = True
    custom_config.DEFAULT_BATCH_SIZE = 500

    # Kreiraj sistem
    system = FraudDetectionSystem(
        data_path=Path(__file__).parent.parent / 'data' / 'creditcard.csv',
        sample_fraction=custom_config.SAMPLE_FRACTION,
        use_balancing=custom_config.USE_BALANCING,
        config=custom_config
    )

    # Pokreni kompletan pipeline SA WARM-START
    report = system.run_complete_pipeline(
        batch_size=500,
        streaming_delay=0,  # Bez pauze za brže izvršavanje
        save_results=True,
        warmup_samples=2000  # ✅ ARF će prvo naučiti 2000 primera iz RF-a
    )

    print("\n" + "=" * 70)
    print("  FINALNI IZVEŠTAJ")
    print("=" * 70)
    print(f"\nVreme izvršavanja: {report['elapsed_time_seconds']:.2f}s")
    print(f"Inicijalni RF F1-Score: {report['initial_model_results']['f1'] * 100:.2f}%")

    if report['streaming_summary']:
        totals = report['streaming_summary']['totals']
        print(f"\nUkupno detektovano prevara: {totals['frauds_detected']}/{totals['frauds_encountered']}")
        print(f"Overall Detection Rate: {totals['overall_detection_rate'] * 100:.2f}%")

        print(f"\n💡 ARF je počeo sa {report['configuration']['warmup_samples']} primera znanja iz RF-a!")