from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


class InitialModel:
    def __init__(self, use_balancing, config):
        self.use_balancing = use_balancing
        self.config = config
        self.model = None

    def train(self, X_train, y_train):
        if self.use_balancing:
            # kreiranje laznih prevara
            over = SMOTE(
                sampling_strategy=self.config.SMOTE_SAMPLING_STRATEGY,
                random_state=self.config.RANDOM_SEED
            )

            # brisanje pravih transakcija
            under = RandomUnderSampler(
                sampling_strategy=self.config.UNDERSAMPLING_STRATEGY,
                random_state=self.config.RANDOM_SEED
            )

            pipeline = ImbPipeline(steps=[('over', over), ('under', under)])
            X_train, y_train = pipeline.fit_resample(X_train, y_train)

        self.model = RandomForestClassifier(
            n_estimators=self.config.RF_N_ESTIMATORS,
            max_depth=self.config.RF_MAX_DEPTH,
            min_samples_split=self.config.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=self.config.RF_MIN_SAMPLES_LEAF,
            class_weight=self.config.RF_CLASS_WEIGHT,
            random_state=self.config.RANDOM_SEED,
            n_jobs=self.config.RF_N_JOBS
        )

        self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not initialized")
        return self.model.predict(X)
