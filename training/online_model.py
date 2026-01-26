from river import forest, metrics
from config import Config

class OnlineModel:
    def __init__(self, config=None):
        self.config = config or Config()
        self.model = None

        # Online metrike koje se ažuriraju sa svakom transakcijom
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1 = metrics.F1()
        self.auc = metrics.ROCAUC()

    def initialize(self):
        print("\n" + "=" * 60)
        print("INICIJALIZACIJA ONLINE MODELA (Adaptive Random Forest)")
        print("=" * 60)
        print(f"Broj stabala: {self.config.ARF_N_MODELS}")
        print(f"Max features: {self.config.ARF_MAX_FEATURES}")
        print(f"Lambda: {self.config.ARF_LAMBDA}")

        self.model = forest.ARFClassifier(
            n_models=self.config.ARF_N_MODELS,
            max_features=self.config.ARF_MAX_FEATURES,
            lambda_value=self.config.ARF_LAMBDA,
            seed=self.config.RANDOM_SEED
        )

        print("✓ Online model kreiran!")
        return self.model

    def learn_one(self, x_dict, y_true):
        """
        Uči na jednom primeru (online learning).

        Args:
            x_dict (dict): Feature-i kao dictionary {feature_name: value}
            y_true (bool): Prava labela (True=prevara, False=legitimna)
        """
        if self.model is None:
            raise ValueError("Model nije inicijalizovan! Pozovi initialize()")

        # Prvo predvidi (da bi evaluirali pre učenja)
        y_pred = self.predict_one(x_dict)
        y_pred_proba = self.predict_proba_one(x_dict).get(True, 0)

        # Ažuriraj metrike
        self.accuracy.update(y_true, y_pred)
        self.precision.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        self.f1.update(y_true, y_pred)
        self.auc.update(y_true, y_pred_proba)

        # Sada uči na ovom primeru
        self.model.learn_one(x_dict, y_true)

    def predict_one(self, x_dict):
        """
        Predikcija na jednom primeru.

        Args:
            x_dict (dict): Feature-i

        Returns:
            bool: Predikcija (True=prevara, False=legitimna)
        """
        if self.model is None:
            raise ValueError("Model nije inicijalizovan!")

        proba = self.model.predict_proba_one(x_dict)
        return proba.get(True, 0) > 0.5

    def predict_proba_one(self, x_dict):
        """
        Verovatnoće predikcija na jednom primeru.

        Args:
            x_dict (dict): Feature-i

        Returns:
            dict: {False: prob_legitimate, True: prob_fraud}
        """
        if self.model is None:
            raise ValueError("Model nije inicijalizovan!")

        return self.model.predict_proba_one(x_dict)

    def get_metrics(self):
        """
        Vraća trenutne vrednosti svih metrika.

        Returns:
            dict: Dictionary sa metrikama
        """
        return {
            'accuracy': self.accuracy.get(),
            'precision': self.precision.get(),
            'recall': self.recall.get(),
            'f1': self.f1.get(),
            'auc': self.auc.get()
        }

    def reset_metrics(self):
        """Resetuje sve metrike (korisno za novi batch)."""
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1 = metrics.F1()
        self.auc = metrics.ROCAUC()