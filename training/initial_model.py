from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

class InitialModel:
    def __init__(self, use_balancing, config):
        self.use_balancing = use_balancing
        self.config = config
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        print("\n" + "=" * 60)
        print("TRENIRANJE INICIJALNOG RANDOM FOREST MODELA")
        print("=" * 60)

        # Prikaži distribuciju klasa pre balansiranja
        fraud_count = sum(y_train)
        total_count = len(y_train)
        print(f"\nOriginalna distribucija:")
        print(f"  Legitimne transakcije: {total_count - fraud_count:,} ({(1 - fraud_count / total_count) * 100:.2f}%)")
        print(f"  Prevare: {fraud_count:,} ({fraud_count / total_count * 100:.2f}%)")

        if self.use_balancing:
            print(f"\n{'=' * 60}")
            print("BALANSIRANJE PODATAKA (SMOTE + Undersampling)")
            print("=" * 60)

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

            # Prikaži novu distribuciju
            fraud_count_new = sum(y_train_resampled)
            total_count_new = len(y_train_resampled)
            print(f"\nNova distribucija:")
            print(
                f"  Legitimne transakcije: {total_count_new - fraud_count_new:,} ({(1 - fraud_count_new / total_count_new) * 100:.2f}%)")
            print(f"  Prevare: {fraud_count_new:,} ({fraud_count_new / total_count_new * 100:.2f}%)")

            X_train_final = X_train_resampled
            y_train_final = y_train_resampled
        else:
            print("\n⚠️  BALANSIRANJE ISKLJUČENO - koriste se originalni podaci")
            X_train_final = X_train
            y_train_final = y_train

        # KREIRANJE I TRENIRANJE RANDOM FOREST MODELA
        print(f"\n{'=' * 60}")
        print("TRENIRANJE RANDOM FOREST MODELA")
        print("=" * 60)
        print(f"Broj stabala: {self.config.RF_N_ESTIMATORS}")
        print(f"Maksimalna dubina: {self.config.RF_MAX_DEPTH}")
        print(f"Class weight: {self.config.RF_CLASS_WEIGHT}")

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

        print(f"\n{'=' * 60}")
        print("EVALUACIJA NA VALIDATION SETU")
        print("=" * 60)

        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]

        # Classification report
        print("\nDetaljni izveštaj:")
        print(classification_report(
            y_val, y_pred,
            target_names=['Legitimna', 'Prevara'],
            digits=4
        ))

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\nConfusion Matrix:")
        print(f"                    Predicted")
        print(f"                Legit    Fraud")
        print(f"Actual  Legit   {tn:6d}   {fp:6d}")
        print(f"        Fraud   {fn:6d}   {tp:6d}")

        # Dodatne metrike
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        auc = roc_auc_score(y_val, y_pred_proba)

        print(f"\nKljučne metrike:")
        print(f"  Accuracy:  {accuracy * 100:.2f}%")
        print(f"  Precision: {precision * 100:.2f}%")
        print(f"  Recall:    {recall * 100:.2f}%")
        print(f"  F1-Score:  {f1 * 100:.2f}%")
        print(f"  ROC-AUC:   {auc:.4f}")

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

    def save(self, filepath):
        if self.model is None:
            raise ValueError("Model nije istreniran!")
        joblib.dump(self.model, filepath)
        print(f"\n✓ Model sačuvan: {filepath}")

    def load(self, filepath):
        self.model = joblib.load(filepath)
        print(f"✓ Model učitan: {filepath}")