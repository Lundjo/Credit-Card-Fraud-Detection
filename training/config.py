class Config:

    SAMPLE_FRACTION = 1.0

    INITIAL_SPLIT = 0.7

    # === BALANSIRANJE PODATAKA ===
    # Da li koristiti balansiranje (SMOTE + Undersampling)?
    USE_BALANCING = True

    # SMOTE parametri - kreiranje sintetičkih primera manjinske klase
    SMOTE_SAMPLING_STRATEGY = 0.3  # Povećaj prevare na 30% većinske klase

    # Undersampling parametri - smanjivanje većinske klase
    UNDERSAMPLING_STRATEGY = 0.7  # Nakon SMOTE, dovedi na 70:30 ratio

    # === RANDOM FOREST (Inicijalni model) ===
    RF_N_ESTIMATORS = 100  # Broj stabala u šumi
    RF_MAX_DEPTH = 20  # Maksimalna dubina stabala
    RF_MIN_SAMPLES_SPLIT = 10  # Min. uzoraka za split node-a
    RF_MIN_SAMPLES_LEAF = 4  # Min. uzoraka u leaf node-u
    RF_CLASS_WEIGHT = 'balanced'  # Automatsko ponderisanje klasa
    RF_N_JOBS = -1  # Koristi sve CPU core-ove

    # === ADAPTIVE RANDOM FOREST (Online model) ===
    ARF_N_MODELS = 10  # Broj stabala u online šumi
    ARF_MAX_FEATURES = 'sqrt'  # Broj feature-a po stablu
    ARF_LAMBDA = 6  # Parametar za Poisson distribuciju

    # === STREAMING SIMULACIJA ===
    DEFAULT_BATCH_SIZE = 1000  # Koliko transakcija procesirati odjednom
    FRAUD_BUFFER_SIZE = 100  # Čuvaj poslednjih N prevara

    # === API ===
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    API_DEBUG = True

    # === SEED za reproduktivnost ===
    RANDOM_SEED = 42