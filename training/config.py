class Config:

    SAMPLE_FRACTION = 1.0
    INITIAL_SPLIT = 0.7

    USE_BALANCING = True
    # kreiranje laznih prevara
    SMOTE_SAMPLING_STRATEGY = 0.3  # 30% od broja legitimnih transakcija
    # smanjivanje legitimnih transakcija
    UNDERSAMPLING_STRATEGY = 0.7  # 70% od novih laznih transakcija

    # inicijalni model
    RF_N_ESTIMATORS = 100  # broj stabala
    RF_MAX_DEPTH = 20  # maksimalna dubina stabala
    RF_MIN_SAMPLES_SPLIT = 10  # minimalno uzoraka za deljenje cvora
    RF_MIN_SAMPLES_LEAF = 4  # minimalno uzoraka u listu
    RF_CLASS_WEIGHT = 'balanced'  # tezinski faktor
    RF_N_JOBS = -1  # sva procesorska jezgra

    # online model
    ARF_N_MODELS = 10  # broj stabala
    ARF_MAX_FEATURES = 'sqrt'  # broj karakteristika
    ARF_LAMBDA = 6  # zamena stabala novim

    # streaming
    DEFAULT_BATCH_SIZE = 500

    # SEED za reproduktivnost
    RANDOM_SEED = 42