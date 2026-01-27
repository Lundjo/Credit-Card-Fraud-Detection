class Config:

    SAMPLE_FRACTION = 1.0
    INITIAL_SPLIT = 0.7

    USE_BALANCING = True
    # kreiranje laznih prevara
    SMOTE_SAMPLING_STRATEGY = 0.3  # 30% od broja legitimnih transakcija
    # smanjivanje legitimnih transakcija
    UNDERSAMPLING_STRATEGY = 0.7  # 70% od novih laznih transakcija

    # Inicijalni model
    RF_N_ESTIMATORS = 100  # Broj stabala u šumi
    RF_MAX_DEPTH = 20  # Maksimalna dubina stabala
    RF_MIN_SAMPLES_SPLIT = 10  # Min. uzoraka za split node-a
    RF_MIN_SAMPLES_LEAF = 4  # Min. uzoraka u leaf node-u
    RF_CLASS_WEIGHT = 'balanced'  # tezinski faktor
    RF_N_JOBS = -1  # Koristi sve CPU core-ove

    # online model
    ARF_N_MODELS = 10  # Broj stabala u online šumi
    ARF_MAX_FEATURES = 'sqrt'  # Broj feature-a po stablu
    ARF_LAMBDA = 6  # zamena stabala novim

    # streaming
    DEFAULT_BATCH_SIZE = 500  # Koliko transakcija procesirati odjednom
    FRAUD_BUFFER_SIZE = 100  # Čuvaj poslednjih N prevara

    # SEED za reproduktivnost
    RANDOM_SEED = 42