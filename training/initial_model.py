from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


class InitialModel:
    def __init__(self, use_balancing, config):
        self.use_balancing = use_balancing
        self.config = config
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
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
            X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)

            X_train_final = X_train_resampled
            y_train_final = y_train_resampled

        else:
            X_train_final = X_train
            y_train_final = y_train

        self.model = RandomForestClassifier(
            n_estimators=self.config.RF_N_ESTIMATORS,
            max_depth=self.config.RF_MAX_DEPTH,
            min_samples_split=self.config.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=self.config.RF_MIN_SAMPLES_LEAF,
            class_weight=self.config.RF_CLASS_WEIGHT,
            random_state=self.config.RANDOM_SEED,
            n_jobs=self.config.RF_N_JOBS
        )

        self.model.fit(X_train_final, y_train_final)

        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Dodatne metrike
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        auc = roc_auc_score(y_val, y_pred_proba)

        # vrati rezultate
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist()
        }

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model nije istreniran!")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model nije istreniran!")
        return self.model.predict_proba(X)
